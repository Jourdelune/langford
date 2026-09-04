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
IMAGE  = "nvidia/cuda:12.8.1-devel-ubuntu22.04"   # sm_120 exige CUDA >= 12.8
SEND   = ["langford6.cu", "build.sh", "run_shard.sh", "run_batch.sh"]
LOCK   = threading.Lock()
STOP   = threading.Event()

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# --------------------------------------------------------------- vast.ai API
def api_key():
    for line in open(os.path.join(HERE, ".env")):
        if line.startswith("VAST_API_KEY="): return line.split("=", 1)[1].strip()
    sys.exit(".env : VAST_API_KEY absent")

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
            sys.exit("\n*** La cle API n'a pas les privileges de gestion d'instances.\n"
                     "    vast.ai exige une cle creee depuis une session 2FA.\n"
                     "    cloud.vast.ai -> Account -> API Keys -> creer une nouvelle cle,\n"
                     "    puis la remplacer dans .env.  Le reste (init/run local/merge)\n"
                     "    fonctionne sans cle.\n")
        raise SystemExit(f"API {method} {path} -> {e.code} {txt}")

def offers(kind="on-demand", limit=40, gpu="RTX 5090"):
    q = {"gpu_name": {"eq": gpu}, "num_gpus": {"eq": 1}, "rentable": {"eq": True},
         "cuda_max_good": {"gte": 12.8}, "reliability2": {"gte": 0.95},
         "inet_down": {"gte": 100}, "type": kind,
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
            "AND (status='pending' OR lease<?) ORDER BY tries, RANDOM() LIMIT ?",
            (now, k)).fetchall()
        ids = [r[0] for r in rows]
        if ids:
            c.executemany("UPDATE tasks SET status='leased', lease=?, worker=?, "
                          "tries=tries+1 WHERE id=?", [(now + secs, worker, i) for i in ids])
        c.execute("COMMIT")
    return ids

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
    subprocess.run(["scp", "-i", KEYF, "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                    "-P", str(w["port"])] + [os.path.join(HERE, f) for f in SEND] +
                   [f"root@{w['host']}:/root/"], check=True, timeout=300)
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
        o = offers(kind, limit=a.count)
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
    print(f"  une 5090 en fournit {a.ratio:.2f}/h -> {a.ratio*a.hours:.1f} h GPU en {a.hours} h")
    print(f"\n  => {k} RTX 5090")
    for p, lbl in ((0.20, "spot"), (0.35, "a la demande")):
        print(f"     {lbl:<14} {k} x {a.hours} h x {p:.2f} $ = {k*a.hours*p:6.2f} $")
    print(f"\n  Le rapport {a.ratio} est un MODELE. Le mesurer d'abord :")
    print(f"     ./orchestrator.py up --count 1 && ./orchestrator.py bench")

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
    offs = offers(kind, limit=300)
    if not offs: sys.exit("aucune offre 5090 disponible")
    c = db(); made = 0
    seen = {r[0] for r in c.execute("SELECT inst FROM workers WHERE inst IS NOT NULL")}
    for off in offs:
        if made >= a.count: break
        if a.max_price and off["dph_total"] > a.max_price: continue
        body = {"client_id": "me", "image": IMAGE, "disk": 12, "runtype": "ssh",
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
        c.execute("INSERT OR REPLACE INTO workers(name,kind,inst,price,state,seen) "
                  "VALUES(?,'ssh',?,?,'new',?)", (name, iid, off["dph_total"], time.time()))
        log(f"loue {name}  {off['dph_total']:.3f} $/h  {off.get('geolocation')}")
        made += 1
    log(f"{made} instance(s) creee(s)")

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
    row = c.execute("SELECT AVG(secs) FROM (SELECT secs FROM tasks WHERE status='done' "
                    "ORDER BY rowid DESC LIMIT 200)").fetchone()[0]
    rate = c.execute("SELECT SUM(1.0/spv) FROM workers WHERE spv>0 AND state='running'").fetchone()[0]
    eta = ""
    if rate and d:
        left = spv_from(row, n, T) * (tot - d) / rate if row else 0
        eta = f"   ETA {left/3600:.1f} h"
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
    q = S.add_parser("plan");   q.add_argument("--hours", type=float, default=10); q.add_argument("--ratio", type=float, default=4.02); q.add_argument("--base", type=float, default=772); q.set_defaults(f=cmd_plan)
    q = S.add_parser("offers"); q.add_argument("--count", type=int, default=12); q.add_argument("--bid", action="store_true"); q.set_defaults(f=cmd_offers)
    q = S.add_parser("up");     q.add_argument("--count", type=int, required=True); q.add_argument("--bid", type=float, default=0); q.add_argument("--max-price", type=float, default=0.40); q.set_defaults(f=cmd_up)
    q = S.add_parser("sshkey"); q.set_defaults(f=cmd_sshkey)
    q = S.add_parser("add");    q.add_argument("--ssh"); q.add_argument("--host"); q.add_argument("--port", type=int); q.add_argument("--name"); q.add_argument("--price", type=float, default=0.35); q.set_defaults(f=cmd_add)
    q = S.add_parser("bench"); q.add_argument("--samples", type=int, default=96); q.set_defaults(f=cmd_bench)
    q = S.add_parser("run");    q.add_argument("--local-only", action="store_true"); q.set_defaults(f=cmd_run)
    q = S.add_parser("status"); q.set_defaults(f=cmd_status)
    q = S.add_parser("down");   q.set_defaults(f=cmd_down)
    q = S.add_parser("merge");  q.set_defaults(f=cmd_merge)
    a = P.parse_args(); a.f(a)

if __name__ == "__main__": main()
