#!/usr/bin/env python3
"""
Orchestrateur pour le calcul de L(2,n) reparti sur des GPU loues chez vast.ai.

Principes de conception, et pourquoi :

  * Le maitre (le PC a la 4070, allume en permanence) initie TOUTES les
    connexions.  Une machine domestique est derriere un NAT : les instances
    louees ne peuvent pas l'appeler.  Le maitre pousse donc le travail par SSH.

  * L'etat vit dans SQLite en mode WAL, et chaque resultat est valide des son
    arrivee.  Tuer le maitre au milieu du calcul ne coute que les lots en vol :
    au redemarrage, les baux expires repassent en attente et on repart de la.

  * Le travail est distribue par BAUX, pas par tranches fixes.  Un worker
    demande un lot, on le lui reserve pour une duree bornee ; s'il ne rend rien
    a temps (preemption spot, machine perdue, SSH coupe), le bail expire et le
    lot retourne au pot.  Aucune tache ne peut donc etre definitivement perdue,
    et les workers lents ne bloquent pas les rapides.

  * Les taches sont a charge egale (cf. run_shard.sh), et le lot est dimensionne
    d'apres le debit mesure de chaque worker pour durer ~6 min.  Une coupure ne
    perd donc jamais plus que quelques minutes de calcul.

  * La 4070 locale est un worker comme les autres, simplement sans SSH.

Usage :
    ./orchestrator.py init   -n 31 -T 8192
    ./orchestrator.py plan   --hours 10
    ./orchestrator.py offers
    ./orchestrator.py up     --count 20 [--bid 0.25]
    ./orchestrator.py run                       # boucle principale, reprenable
    ./orchestrator.py status
    ./orchestrator.py down
    ./orchestrator.py merge
"""
import argparse, json, os, queue, random, re, sqlite3, subprocess, sys, threading, time, urllib.error, urllib.request

HERE   = os.path.dirname(os.path.abspath(__file__))
DB     = os.path.join(HERE, "state.db")
KEYF   = os.path.join(HERE, ".ssh_orch")
API    = "https://console.vast.ai/api/v0/"
# Image par defaut : `base` (~200 Mo) et non `devel` (~6 Go).  Le binaire
# prefabrique `langford6.fat` est lie en STATIQUE (`-cudart static`) et couvre
# sm_89 + sm_120 : il n'a besoin que du pilote, pas du toolkit.  Mesure : la
# `devel` met plus de 25 min a se telecharger sur certains hotes -- plus que le
# calcul lui-meme.  `--image devel` remet l'ancienne si l'on veut compiler sur
# place (indispensable pour une architecture non couverte par le .fat).
# ubuntu24.04 et non 22.04 : le .fat est compile sur une machine a glibc
# recente, et l'image 22.04 (glibc 2.35) le refuse -- "GLIBC_2.38 not found".
IMAGE_BASE  = "nvidia/cuda:12.8.1-base-ubuntu24.04"
IMAGE_DEVEL = "nvidia/cuda:12.8.1-devel-ubuntu24.04"   # sm_120 exige CUDA >= 12.8
IMAGE  = IMAGE_BASE
SEND   = ["langford6.cu", "build.sh", "run_shard.sh", "run_batch.sh"]
LOCK   = threading.Lock()
STOP   = threading.Event()

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# --------------------------------------------------------------- vast.ai API
def env_get(k):
    try: f = open(os.path.join(HERE, ".env"))
    except FileNotFoundError: return None
    for line in f:
        if line.startswith(k + "="): return line.split("=", 1)[1].strip()
    return None

def env_set(k, v):
    p = os.path.join(HERE, ".env")
    lines = [l for l in (open(p).read().splitlines() if os.path.exists(p) else [])
             if not l.startswith(k + "=")]
    lines.append(f"{k}={v}")
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f: f.write("\n".join(lines) + "\n")

def api_key():
    """Sur un compte protege par 2FA, la cle API ne donne AUCUN privilege a elle
    seule : il faut l'echanger contre une session_key en presentant un code TOTP
    (cf. cmd_tfa). C'est cette session_key qu'on utilise si elle existe."""
    k = env_get("VAST_SESSION_KEY") or env_get("VAST_API_KEY")
    if not k: sys.exit(".env : ni VAST_SESSION_KEY ni VAST_API_KEY")
    return k

def api(method, path, body=None, auth=True):
    req = urllib.request.Request(API + path, method=method,
        data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    if auth: req.add_header("Authorization", "Bearer " + api_key())
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return json.load(r)
    except urllib.error.HTTPError as e:
        txt = e.read().decode()[:300]
        if e.code == 401 and "Two Factor" in txt:
            sys.exit(
                "\n*** Session 2FA requise.  Sur un compte protege par 2FA, la cle\n"
                "    API ne donne aucun privilege a elle seule : il faut l'echanger\n"
                "    contre une session_key en presentant un code TOTP.\n\n"
                "        ./orchestrator.py tfa --code 123456\n\n"
                "    (code a 6 chiffres de l'application d'authentification, ou\n"
                "     ./orchestrator.py tfa --backup ABCD-EFGH-IJKL)\n\n"
                "*** ancien diagnostic, conserve pour memoire :\n"
                "    Constate sur instances/, machines/, invoices/, users/current/,\n"
                "    team/members/ ET tfa/status/ : erreur identique partout, avec\n"
                "    trois cles differentes. Ce n'est donc ni la cle ni son scope,\n"
                "    mais une barriere au niveau du compte -- et il n'existe aucun\n"
                "    endpoint permettant d'elever une cle avec un code TOTP.\n\n"
                "    A faire dans la console (cloud.vast.ai) :\n"
                "      1. se deconnecter, se reconnecter, et VERIFIER qu'un code a\n"
                "         6 chiffres est demande. Sinon la 2FA n'est pas active,\n"
                "         et c'est la cause.\n"
                "      2. depuis cette session-la : Keys -> +New (full access).\n\n"
                "    L'API ne sert qu'a `up` et `down`. Pour lancer sans elle :\n"
                "      ./orchestrator.py sshkey     (a coller dans Account -> SSH Keys)\n"
                "      ./orchestrator.py add --ssh \"ssh -p PORT root@HOTE\"\n"
                "      ./orchestrator.py run\n")
        raise SystemExit(f"API {method} {path} -> {e.code} {txt}")

def offers(kind="on-demand", limit=40, gpu="RTX 5090", min_net=100, verified=False):
    q = {"gpu_name": {"eq": gpu}, "num_gpus": {"eq": 1}, "rentable": {"eq": True},
         "cuda_max_good": {"gte": 12.8}, "reliability2": {"gte": 0.95},
         "inet_down": {"gte": min_net}, "type": kind,
         # Les offres les moins cheres sont systematiquement "deverified", et
         # c'est la ou se trouvent les cartes sur-souscrites : une 4090 a 0,136
         # portait cinq autres locataires. Une 4090 verifiee coute 0,309, soit
         # le prix d'une 5090 -- le rabais etait le symptome.
         **({"verified": {"eq": True}} if verified else {}),
         "order": [["dph_total", "asc"]], "limit": limit}
    return api("PUT", "search/asks/", {"q": q}, auth=False).get("offers", [])

# ------------------------------------------------------------------- etat
def db():
    c = sqlite3.connect(DB, timeout=60, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")      # survit a un kill -9 du maitre
    c.execute("PRAGMA synchronous=FULL")      # ... et a une coupure de courant
    return c

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'pending',
  lease REAL DEFAULT 0, worker TEXT, part TEXT, secs REAL, tries INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS i_status ON tasks(status, lease);
CREATE TABLE IF NOT EXISTS workers(
  name TEXT PRIMARY KEY, kind TEXT, inst INTEGER, host TEXT, port INTEGER,
  price REAL, state TEXT, seen REAL, ndone INTEGER DEFAULT 0, spv REAL);
"""

def meta(c, k, v=None):
    if v is None:
        r = c.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else None
    c.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=?", (k, str(v), str(v)))

def cmd_init(a):
    c = db(); c.executescript(SCHEMA)
    meta(c, "n", a.n); meta(c, "T", a.T)
    c.execute("BEGIN")
    c.executemany("INSERT OR IGNORE INTO tasks(id) VALUES(?)",
                  [(str(k),) for k in range(a.T)] + [("D",)])
    c.execute("COMMIT")
    log(f"n={a.n}  {a.T} taches + diagonale  ->  {DB}")

# ---------------------------------------------------------------- dispatch
def per_task(spv, n, T):
    """Duree d'une tache : les taches sont a charge egale, donc c'est le cout
    total (spv secondes par vhi en moyenne uniforme, x 2^(n-8) valeurs) / T."""
    return spv * (2 ** (n - 8)) / T

def spv_from(sec_per_task, n, T):
    return sec_per_task * T / (2 ** (n - 8))

def lease(c, worker, k, secs):
    """Reserve jusqu'a k taches. Atomique : deux workers ne peuvent pas obtenir
    la meme tache, meme si le maitre tourne en plusieurs fils."""
    now = time.time()
    with LOCK:
        c.execute("BEGIN IMMEDIATE")
        rows = c.execute(
            "SELECT id FROM tasks WHERE status!='done' "
            # Les taches de petit indice sont les plus lourdes (la loi
            # d'equilibrage les sous-estime : vhi=0 coute 2,09 s/vhi mesure
            # contre 0,66 predit). Les servir en premier, c'est la regle LPT,
            # qui minimise la duree totale sur machines paralleles : une tache
            # longue tiree en fin de course allongerait la queue pour tout le
            # monde. RANDOM() faisait exactement l'inverse.
            "AND (status='pending' OR lease<?) ORDER BY tries, CAST(id AS INTEGER) LIMIT ?",
            (now, k)).fetchall()
        ids = [r[0] for r in rows]
        if ids:
            c.executemany("UPDATE tasks SET status='leased', lease=?, worker=?, "
                          "tries=tries+1 WHERE id=?", [(now + secs, worker, i) for i in ids])
        c.execute("COMMIT")
    return ids

def part_ok(part, n):
    """Auto-test par tache.  Chaque terme de la somme est un produit des n
    facteurs A_i (i = 2..n+1) et A_i = i (mod 2), donc les floor((n+1)/2)
    ecarts PAIRS donnent chacun un facteur 2 : toute somme partielle est
    divisible par 2^E(n).  Une tranche corrompue est ainsi rejetee a l'arrivee
    -- son bail expire et elle repart au pot -- au lieu de n'etre vue qu'a
    l'agregation finale, apres des semaines de calcul."""
    try:
        v = int(part.replace(":", ""), 16)
    except ValueError:
        return False
    return len(part) == 44 and v % (1 << ((n + 1) // 2)) == 0

def finish(c, tid, part, secs, worker):
    with LOCK:
        c.execute("UPDATE tasks SET status='done', part=?, secs=?, worker=? WHERE id=?",
                  (part, secs, worker, tid))
        c.execute("UPDATE workers SET ndone=ndone+1, seen=? WHERE name=?", (time.time(), worker))

def remaining(c):
    return c.execute("SELECT COUNT(*) FROM tasks WHERE status!='done'").fetchone()[0]

# ------------------------------------------------------------------ workers
SSH = ["ssh", "-i", KEYF, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
       "-o", "LogLevel=ERROR", "-o", "BatchMode=yes"]

def ssh_cmd(w, cmd, timeout=None):
    return subprocess.run(SSH + ["-p", str(w["port"]), f"root@{w['host']}", cmd],
                          capture_output=True, text=True, timeout=timeout)

def provision(w):
    """Attend le SSH, envoie les sources, compile. Idempotent : relancable."""
    for _ in range(60):
        if STOP.is_set(): return False
        r = ssh_cmd(w, "echo ok", timeout=30)
        if r.returncode == 0 and "ok" in r.stdout: break
        time.sleep(10)
    else:
        log(f"{w['name']} : SSH injoignable, abandon"); return False
    # Verifier que la carte est bien A NOUS. Une offre bon marche peut etre une
    # carte sur-souscrite : une "RTX 4090 a 0,136 $/h" testee le 2026-09-04
    # portait cinq autres processus sur le meme UUID, 20 Go occupes et 100 %
    # d'utilisation avant meme qu'on lance quoi que ce soit -- elle mesurait
    # 0,85x une 4070 au lieu de 2,8x. Refuser la machine coute moins cher que
    # de payer un sixieme de GPU pendant dix heures.
    r = ssh_cmd(w, "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits", timeout=60)
    try:
        mem, util = (int(x.strip()) for x in r.stdout.strip().split(",")[:2])
        if mem > 2000 or util > 25:
            log(f"{w['name']} : carte deja occupee ({mem} Mo, {util} %) -- ecartee")
            return False
    except Exception:
        log(f"{w['name']} : etat GPU illisible, on continue")

    files = [os.path.join(HERE, f) for f in SEND]
    fat = os.path.join(HERE, "langford6.fat")
    if os.path.exists(fat): files.append(fat)
    subprocess.run(["scp", "-i", KEYF, "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                    "-P", str(w["port"])] + files +
                   [f"root@{w['host']}:/root/"], check=True, timeout=600)
    # On prefere un binaire prefabrique couvrant sm_89 et sm_120 : 2 Mo, cudart
    # lie en statique, aucune dependance CUDA dynamique. Cela evite une minute
    # de nvcc facturee sur chaque instance -- et permettrait a terme une image
    # `base` de 200 Mo au lieu de la `devel` de 6 Go. On verifie qu'il tourne
    # vraiment sur la carte distante ; sinon on retombe sur la compilation.
    if os.path.exists(fat):
        r = ssh_cmd(w, "cd /root && chmod +x *.sh langford6.fat && "
                       "mv -f langford6.fat langford6 && ./langford6 -n 12 2>/dev/null | tail -1",
                    timeout=180)
        if "108144" in r.stdout:          # L(2,12) : le binaire calcule juste ici
            log(f"{w['name']} : binaire prefabrique operationnel"); return True
        log(f"{w['name']} : binaire prefabrique inutilisable, compilation")
    r = ssh_cmd(w, "cd /root && chmod +x *.sh && nvcc -O3 -arch=native -o langford6 langford6.cu 2>&1 | tail -3 && ls -l langford6", timeout=900)
    if r.returncode != 0 or "langford6" not in r.stdout:
        log(f"{w['name']} : compilation echouee : {(r.stdout + r.stderr)[-300:]}"); return False
    return True

def worker_loop(w, n, T):
    """Un fil par worker : bail -> calcul -> validation immediate, en boucle.
    Chaque fil ouvre sa propre connexion : SQLite interdit de la partager."""
    c = db()
    if w["kind"] != "local" and not provision(w): return
    log(f"{w['name']} : pret")
    spv = w.get("spv") or 0.35
    while not STOP.is_set() and remaining(c):
        per   = per_task(spv, n, T)               # duree estimee d'une tache
        batch = max(1, min(64, int(360 / max(per, 1e-9))))
        ids   = lease(c, w["name"], batch, secs=max(900, 4 * batch * per))
        if not ids:
            time.sleep(20); continue
        t0 = time.time(); got = 0
        cmd = f"cd /root && ./run_batch.sh {n} {T} " + " ".join(ids)
        try:
            if w["kind"] == "local":
                p = subprocess.Popen(["./run_batch.sh", str(n), str(T)] + ids, cwd=HERE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
            else:
                p = subprocess.Popen(SSH + ["-p", str(w["port"]), f"root@{w['host']}", cmd],
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
            for line in p.stdout:                      # au fil de l'eau : une
                f = line.split()                       # coupure ne perd que la
                if len(f) == 2 and f[0] != "ERR":      # tache en cours
                    if not part_ok(f[1], n):           # rejetee : le bail
                        log(f"{w['name']} : tache {f[0]} REJETEE ({f[1]})")
                        continue                       # expire, elle repart
                    finish(c, f[0], f[1], (time.time() - t0) / max(got + 1, 1), w["name"])
                    got += 1
            p.wait(timeout=60)
        except Exception as e:
            log(f"{w['name']} : {type(e).__name__} {e}")
        if got:
            spv = 0.7 * spv + 0.3 * spv_from((time.time() - t0) / got, n, T)
            with LOCK: c.execute("UPDATE workers SET spv=?, seen=? WHERE name=?",
                                 (spv, time.time(), w["name"]))
        else:
            log(f"{w['name']} : lot vide, pause"); time.sleep(30)
    log(f"{w['name']} : fini")

# ------------------------------------------------------------------ commandes
def ensure_key():
    if not os.path.exists(KEYF):
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", KEYF,
                        "-C", "langford-orchestrator"], check=True)
        os.chmod(KEYF, 0o600)
    return open(KEYF + ".pub").read().strip()

def cmd_offers(a):
    for kind in (["bid"] if a.bid else ["on-demand", "bid"]):
        o = offers(kind, limit=a.count, gpu=a.gpu, min_net=a.min_net, verified=a.verified)
        print(f"\n=== {kind} : {len(o)} offres ===")
        print(f"{'offre':>10} {'$/h':>6} {'min':>6} {'cuda':>5} {'fiab':>5} {'net':>6}  lieu")
        for x in o[:a.count]:
            print(f"{x['id']:>10} {x['dph_total']:>6.3f} {x.get('min_bid',0):>6.3f} "
                  f"{x.get('cuda_max_good',0):>5} {x.get('reliability2',0):>5.3f} "
                  f"{x.get('inet_down',0):>6.0f}  {x.get('geolocation')}")

def cmd_plan(a):
    c = db(); n = int(meta(c, "n") or 31)
    import math
    base  = float(a.base)                      # heures-GPU, echelle 4070
    rest  = base - a.hours * 1.0               # ce que la 4070 locale absorbe
    k     = max(0, math.ceil(rest / (a.ratio * a.hours)))
    print(f"n={n} : {base:.0f} h GPU (base 4070), cible {a.hours} h")
    print(f"  la 4070 locale en absorbe        {a.hours*1.0:.0f}  ({100*a.hours/base:.1f} %)")
    print(f"  reste a couvrir                  {rest:.0f} h GPU")
    print(f"  une {a.gpu} en fournit {a.ratio:.2f}/h -> {a.ratio*a.hours:.1f} h GPU en {a.hours} h")
    print(f"\n  => {k} {a.gpu}")
    spot, dem = (0.20, 0.35) if "5090" in a.gpu else (0.11, 0.25)
    for p, lbl in ((spot, "spot"), (dem, "a la demande")):
        print(f"     {lbl:<14} {k} x {a.hours} h x {p:.2f} $ = {k*a.hours*p:6.2f} $")
    # Les deux rapports MESURES, avec la version sur laquelle ils l'ont ete.
    # Ne rien affirmer d'autre : un --ratio passe a la main n'est pas une mesure.
    print()
    print(f"  Rapports mesures a ce jour, a travail identique (memes vhi, graine")
    print(f"  fixe, un seul processus par carte) :")
    print(f"    RTX 5090 : 3,05  (mesure le 2026-09-04, sur la v6.1)")
    print(f"    RTX 4090 : 2,66  (mesure le 2026-09-05, sur la v7)")
    if abs(a.ratio - 3.05) > 1e-9 and abs(a.ratio - 2.66) > 1e-9:
        print(f"    -> le rapport {a.ratio} utilise ici n'est AUCUN des deux : c'est une")
        print(f"       hypothese passee en ligne de commande, pas une mesure.")

def cmd_bench(a):
    """Mesure le vrai debit de chaque worker. Le rapport 5090/4070 utilise par
    `plan` est un modele d'architecture ; ceci le remplace par une mesure, sur
    le meme echantillon de vhi des deux cotes (la graine de --bench est fixe)."""
    c = db(); n = int(meta(c, "n") or 31)
    try: refresh(c)
    except SystemExit as e: log(str(e))
    rows = list(c.execute("SELECT name,kind,host,port,state FROM workers"))
    for name, kind, host, port, state in rows:
        w = dict(name=name, kind=kind, host=host, port=port)
        if kind != "local":
            if not host: log(f"{name} : pas d'adresse SSH, ignore"); continue
            if not provision(w): continue
            r = ssh_cmd(w, f"cd /root && ./langford6 -n {n} --bench {a.samples}", timeout=3600)
            out = r.stdout
        else:
            out = subprocess.run([os.path.join(HERE, "langford6"), "-n", str(n),
                                  "--bench", str(a.samples)], capture_output=True,
                                 text=True, cwd=HERE).stdout
        m = re.search(r"BENCH_SPV=([\d.]+)\s+BENCH_GPUH=([\d.]+)", out)
        if not m: log(f"{name} : pas de mesure ({out[-160:]})"); continue
        spv, gpuh = float(m.group(1)), float(m.group(2))
        c.execute("UPDATE workers SET spv=? WHERE name=?", (spv, name))
        gpu = re.search(r"GPU\s+:\s+(.+?)\s\s", out)
        log(f"{name:<14}{(gpu.group(1) if gpu else '?'):<26}{spv:.4f} s/vhi   {gpuh:7.1f} h GPU pour n={n}")
    base = c.execute("SELECT spv FROM workers WHERE name='local'").fetchone()
    if base and base[0]:
        print(f"\nrapports mesures (base = 4070 locale) :")
        for name, spv in c.execute("SELECT name,spv FROM workers WHERE spv>0 AND name!='local'"):
            print(f"  {name:<14} x{base[0]/spv:.2f}")

def cmd_up(a):
    pub = ensure_key()
    onstart = ("mkdir -p /root/.ssh && echo '%s' >> /root/.ssh/authorized_keys && "
               "chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys" % pub)
    kind = "bid" if a.bid else "on-demand"
    offs = offers(kind, limit=300, gpu=a.gpu, min_net=a.min_net, verified=a.verified)
    if not offs: sys.exit("aucune offre 5090 disponible")
    c = db(); made = 0
    seen = {r[0] for r in c.execute("SELECT inst FROM workers WHERE inst IS NOT NULL")}
    for off in offs:
        if made >= a.count: break
        if a.max_price and off["dph_total"] > a.max_price: continue
        img = IMAGE_DEVEL if getattr(a, "image", "base") == "devel" else IMAGE_BASE
        body = {"client_id": "me", "image": img, "disk": 12, "runtype": "ssh",
                "onstart": onstart, "env": {}}
        if a.bid: body["price"] = round(max(off.get("min_bid", 0) * 1.05, a.bid), 4)
        try: r = api("PUT", f"asks/{off['id']}/", body)
        except SystemExit as e:
            if "Two Factor" in str(e): raise
            continue
        if not r.get("success"): continue
        iid = r.get("new_contract")
        if iid in seen: continue
        name = f"vast{iid}"
        # Louer ne DEMARRE pas : l'instance nait avec intended_status=stopped,
        # elle telecharge l'image puis reste la, et son port SSH refuse la
        # connexion indefiniment.  Mesure du 2026-09-05 : trois locations
        # perdues a attendre un SSH qui ne viendrait jamais.  On demande donc
        # explicitement le demarrage -- apres quoi le SSH repond en secondes.
        try:
            api("PUT", f"instances/{iid}/", {"state": "running"})
        except SystemExit as e:
            log(f"{name} : demarrage refuse ({e})")
        c.execute("INSERT OR REPLACE INTO workers(name,kind,inst,price,state,seen) "
                  "VALUES(?,'ssh',?,?,'new',?)", (name, iid, off["dph_total"], time.time()))
        log(f"loue {name}  {off['dph_total']:.3f} $/h  {off.get('geolocation')}  (demarrage demande)")
        made += 1
    log(f"{made} instance(s) creee(s)")

def cmd_tfa(a):
    """Echange un code a 6 chiffres contre une session_key elevee.
    Un code ne vaut que ~30 s : avoir la commande prete evite de le rater.

    --send sms|email demande d'abord l'envoi du code et affiche le `secret`
    qu'il faut ensuite repasser avec le code (le TOTP, lui, n'en a pas besoin)."""
    if a.send:
        r = api("POST", f"tfa/{a.send}/", {})
        sec = r.get("secret") or r.get("tfa_secret")
        log(f"code envoye par {a.send}. Puis :")
        log(f"  ./orchestrator.py tfa --method {a.send} --secret {sec} --code <CODE>")
        return
    if a.backup:      body = {"backup_code": a.backup}
    else:             body = {"tfa_method": a.method, "code": str(a.code)}
    if a.secret:      body["secret"] = a.secret
    if a.method_id:   body["tfa_method_id"] = a.method_id
    r = api("POST", "tfa/", body)
    sk = r.get("session_key")
    if not sk: sys.exit(f"pas de session_key dans la reponse : {str(r)[:200]}")
    env_set("VAST_SESSION_KEY", sk)
    log("session 2FA obtenue, ecrite dans .env (VAST_SESSION_KEY)")
    left = r.get("backup_codes_remaining")
    if left is not None: log(f"codes de secours restants : {left}")
    try:
        n = len(api("GET", "instances/").get("instances", []))
        log(f"verification : l'API repond, {n} instance(s) sur le compte")
    except SystemExit as e: log(f"verification : {e}")

def cmd_sshkey(a):
    pub = ensure_key()
    print("Cle publique de l'orchestrateur -- a coller dans la console vast.ai\n"
          "(Account -> SSH Keys -> New), puis louer les instances a la main :\n")
    print(pub)

def cmd_add(a):
    """Enregistre une instance louee a la main. Accepte soit --host/--port, soit
    directement la ligne 'Connect' que donne la console vast.ai."""
    host, port = a.host, a.port
    if a.ssh:
        m = re.search(r"-p\s*(\d+).*?root@([\w.\-]+)", a.ssh) or \
            re.search(r"root@([\w.\-]+).*?-p\s*(\d+)", a.ssh)
        if not m: sys.exit("ligne SSH incomprise : attendu 'ssh -p PORT root@HOTE'")
        g = m.groups()
        port, host = (int(g[0]), g[1]) if g[0].isdigit() else (int(g[1]), g[0])
    if not host or not port: sys.exit("il faut --ssh '...' ou --host H --port P")
    c = db(); name = a.name or f"ssh{port}"
    c.execute("INSERT OR REPLACE INTO workers(name,kind,host,port,price,state,seen) "
              "VALUES(?,'ssh',?,?,?,'running',?)", (name, host, port, a.price, time.time()))
    log(f"{name} -> root@{host}:{port}")

def refresh(c):
    """Recolle les adresses SSH depuis l'API et retire les instances mortes."""
    inst = {i["id"]: i for i in api("GET", "instances/").get("instances", [])}
    for name, iid in c.execute("SELECT name,inst FROM workers WHERE inst IS NOT NULL").fetchall():
        i = inst.get(iid)
        if not i:
            c.execute("UPDATE workers SET state='gone' WHERE name=?", (name,)); continue
        c.execute("UPDATE workers SET host=?,port=?,state=? WHERE name=?",
                  (i.get("ssh_host"), i.get("ssh_port"), i.get("actual_status") or "?", name))
    return inst

def cmd_run(a):
    c = db(); n = int(meta(c, "n")); T = int(meta(c, "T"))
    # Verrou : le maitre est le seul distributeur ; deux `run` simultanes
    # remettraient la meme tache en jeu deux fois.
    global _LOCKF
    import fcntl
    _LOCKF = open(os.path.join(HERE, ".run.lock"), "w")
    try: fcntl.flock(_LOCKF, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit("un `run` tourne deja (verrou .run.lock)")
    # Puisqu'il est seul et qu'il vient de demarrer, rien n'est en vol : on
    # libere tout de suite les baux herites du precedent arret au lieu
    # d'attendre leur expiration (sinon, jusqu'a 15 min perdues au redemarrage).
    k = c.execute("UPDATE tasks SET status='pending' WHERE status='leased'").rowcount
    if k: log(f"reprise : {k} tache(s) en vol au dernier arret, remises en attente")
    c.execute("INSERT OR IGNORE INTO workers(name,kind,state,seen) VALUES('local','local','running',?)",
              (time.time(),))
    if not a.local_only:
        try: refresh(c)
        except SystemExit as e: log(str(e)); log("-> on continue avec la seule 4070 locale")
    ws = []
    for r in c.execute("SELECT name,kind,inst,host,port,state,spv FROM workers").fetchall():
        w = dict(zip("name kind inst host port state spv".split(), r))
        if w["kind"] != "local" and (a.local_only or not w["host"]): continue
        ws.append(w)
    log(f"{len(ws)} worker(s) : " + ", ".join(w["name"] for w in ws))
    th = [threading.Thread(target=worker_loop, args=(w, n, T), daemon=True) for w in ws]
    for t in th: t.start()
    try:
        while any(t.is_alive() for t in th) and remaining(c):
            time.sleep(30); show(c, n, T)
    except KeyboardInterrupt:
        log("arret demande ; les baux en cours expireront et seront redistribues")
        STOP.set()
    for t in th: t.join(timeout=10)
    show(c, n, T)

def show(c, n, T):
    d, l, p = (c.execute("SELECT COUNT(*) FROM tasks WHERE status=?", (s,)).fetchone()[0]
               for s in ("done", "leased", "pending"))
    tot = d + l + p
    # Debit agrege : chaque worker fournit 1/per_task(spv) taches par seconde.
    # (L'ancienne formule melangeait des secondes par vhi et des taches ; elle
    #  annoncait 0,4 h la ou il en restait plus de mille.)
    eta = ""
    thr = 0.0
    for (spv,) in c.execute("SELECT spv FROM workers WHERE spv>0 AND state='running'"):
        pt = per_task(spv, n, T)
        if pt > 0: thr += 1.0 / pt
    if thr > 0:
        eta = f"   ETA {(tot - d) / thr / 3600:.1f} h"
    log(f"faites {d}/{tot} ({100*d/tot:.1f} %)  en cours {l}  attente {p}{eta}")

def cmd_status(a):
    c = db(); n = int(meta(c, "n")); T = int(meta(c, "T")); show(c, n, T)
    print(f"\n{'worker':<14}{'etat':<12}{'faites':>7}{'s/vhi':>9}{'$/h':>7}")
    for r in c.execute("SELECT name,state,ndone,spv,price FROM workers ORDER BY ndone DESC"):
        print(f"{r[0]:<14}{str(r[1]):<12}{r[2]:>7}{(r[3] or 0):>9.3f}{(r[4] or 0):>7.3f}")

def cmd_down(a):
    c = db()
    for name, iid in c.execute("SELECT name,inst FROM workers WHERE inst IS NOT NULL AND state!='destroyed'"):
        try:
            api("DELETE", f"instances/{iid}/", {})
            log(f"detruit {name}")
        except SystemExit as e: log(f"{name} : {e}")
        c.execute("UPDATE workers SET state='destroyed' WHERE name=?", (name,))

def cmd_merge(a):
    c = db(); n = int(meta(c, "n"))
    miss = c.execute("SELECT COUNT(*) FROM tasks WHERE status!='done'").fetchone()[0]
    if miss: sys.exit(f"*** {miss} tache(s) non terminee(s) : rien a conclure")
    parts = [r[0] for r in c.execute("SELECT part FROM tasks")]
    f = os.path.join(HERE, "parts_merge.txt")
    open(f, "w").write("\n".join(parts) + "\n")
    os.execv(os.path.join(HERE, "langford6"),
             [os.path.join(HERE, "langford6"), "-n", str(n), "--merge-file", f])

def main():
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S = P.add_subparsers(dest="cmd", required=True)
    q = S.add_parser("init");   q.add_argument("-n", type=int, default=31); q.add_argument("-T", type=int, default=8192); q.set_defaults(f=cmd_init)
    q = S.add_parser("plan");   q.add_argument("--hours", type=float, default=10); q.add_argument("--ratio", type=float, default=2.66); q.add_argument("--base", type=float, default=545)
    q.add_argument("--gpu", default="RTX 4090"); q.set_defaults(f=cmd_plan)
    q = S.add_parser("offers"); q.add_argument("--count", type=int, default=12); q.add_argument("--bid", action="store_true")
    q.add_argument("--gpu", default="RTX 5090"); q.add_argument("--min-net", dest="min_net", type=float, default=100); q.add_argument("--verified", action="store_true"); q.set_defaults(f=cmd_offers)
    q = S.add_parser("up");     q.add_argument("--count", type=int, required=True); q.add_argument("--bid", type=float, default=0); q.add_argument("--max-price", type=float, default=0.40)
    q.add_argument("--image", choices=["base", "devel"], default="base",
                   help="base (200 Mo, exige langford6.fat) ou devel (6 Go, compile sur place)")
    q.add_argument("--gpu", default="RTX 5090"); q.add_argument("--min-net", dest="min_net", type=float, default=400); q.add_argument("--verified", action="store_true"); q.set_defaults(f=cmd_up)
    q = S.add_parser("tfa")
    q.add_argument("--code"); q.add_argument("--backup"); q.add_argument("--secret")
    q.add_argument("--method", default="totp", choices=["totp", "sms", "email"])
    q.add_argument("--method-id", dest="method_id")
    q.add_argument("--send", choices=["sms", "email"], help="faire envoyer le code d'abord")
    q.set_defaults(f=cmd_tfa)
    q = S.add_parser("sshkey"); q.set_defaults(f=cmd_sshkey)
    q = S.add_parser("add");    q.add_argument("--ssh"); q.add_argument("--host"); q.add_argument("--port", type=int); q.add_argument("--name"); q.add_argument("--price", type=float, default=0.35); q.set_defaults(f=cmd_add)
    q = S.add_parser("bench"); q.add_argument("--samples", type=int, default=96); q.set_defaults(f=cmd_bench)
    q = S.add_parser("run");    q.add_argument("--local-only", action="store_true"); q.set_defaults(f=cmd_run)
    q = S.add_parser("status"); q.set_defaults(f=cmd_status)
    q = S.add_parser("down");   q.set_defaults(f=cmd_down)
    q = S.add_parser("merge");  q.set_defaults(f=cmd_merge)
    a = P.parse_args(); a.f(a)

if __name__ == "__main__": main()
