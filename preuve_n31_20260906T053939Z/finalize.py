#!/usr/bin/env python3
"""Cloture du calcul : mettre le resultat a l'abri, couper le loyer, puis prouver.

    ./finalize.py check      etat de preparation, ne touche a rien
    ./finalize.py finish     met a l'abri, detruit les instances, puis audite
    ./finalize.py watch      attend la fin du calcul, puis `finish` tout seul
    ./finalize.py archive [--partiel]   dossier de preuve seul, ne detruit rien

L'ORDRE est le coeur de ce script, et il n'est pas celui qu'on ecrit d'instinct.

Les machines louees ne servent qu'a UNE chose : calculer des taches.  Des que
les 8192 sommes partielles sont dans state.db, elles ne servent plus a rien --
le merge est une addition de 160 bits faite en local, et le recalcul redondant
de l'audit tourne sur la carte LOCALE via run_shard.sh.  Sur la 4070 une tache
coute ~294 s : auditer douze taches avant de couper, c'est une heure de quatre
machines a tourner a vide, ~3 $ jetes pour rien.

D'ou trois phases :

  1. MISE A L'ABRI (secondes, machines encore en vie) -- export des sommes
     partielles hors SQLite, instantane transactionnel de la base, puis merge.
     Le merge refuse si une tranche manque et teste la divisibilite du total
     par 2^(2n+1) : c'est LUI qui attrape une tranche perdue ou corrompue, et
     c'est le seul controle pour lequel garder les machines a un sens -- s'il
     echoue, il faut pouvoir refaire des taches, donc on ne detruit rien.

  2. LIBERATION -- le compteur s'arrete ici, des que le resultat est sur disque
     et arithmetiquement valide.

  3. PREUVE (longue, gratuite) -- audit complet avec recalcul redondant d'un
     echantillon, sources, manifeste, empreintes, archive relue.

Une destruction est irreversible ; une phase 3 ratee ne l'est pas : les sommes
partielles sont deja sur disque, l'audit se relance autant de fois qu'on veut.
"""
import hashlib, importlib.util, json, os, shutil, sqlite3, subprocess, sys, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
DB   = os.path.join(HERE, "state.db")
INST = os.path.join(HERE, ".instances.json")

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def orch():
    spec = importlib.util.spec_from_file_location("orch", os.path.join(HERE, "orchestrator.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, **kw)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def state():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
    n = int(c.execute("SELECT v FROM meta WHERE k='n'").fetchone()[0])
    T = int(c.execute("SELECT v FROM meta WHERE k='T'").fetchone()[0])
    st = dict(c.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status"))
    # Une tache `done` sans somme partielle serait un trou silencieux : le merge
    # la lirait comme une ligne vide et echouerait bien plus loin.  On la compte
    # ici pour pouvoir le dire tout de suite, et surtout pour ne pas couper les
    # machines en croyant le calcul fini.
    creuses = c.execute("SELECT COUNT(*) FROM tasks WHERE status='done' "
                        "AND (part IS NULL OR part='')").fetchone()[0]
    c.close()
    tot = sum(st.values())
    return dict(n=n, T=T, done=st.get("done", 0), total=tot,
                reste=tot - st.get("done", 0), creuses=creuses)

def cmd_check(a=None):
    s = state()
    log(f"n={s['n']}  {s['done']}/{s['total']} taches faites, {s['reste']} restante(s)")
    if s["creuses"]: log(f"*** {s['creuses']} tache(s) 'done' SANS somme partielle")
    ok = s["reste"] == 0 and s["creuses"] == 0
    log("PRET a conclure" if ok else "PAS pret : le calcul n'est pas termine")
    if os.path.exists(INST):
        ins = json.load(open(INST))
        log(f"a liberer le moment venu : {[i['id'] for i in ins]} "
            f"({sum(i['dph'] for i in ins):.3f} $/h)")
    else:
        log("pas de .instances.json : `finish` ne detruira rien (c'est voulu)")
    return ok

# --------------------------------------------- phase 1 : mise a l'abri
def preserve(exiger_complet=True):
    """Tout ce qui doit etre fait AVANT de couper. Quelques secondes."""
    s = state()
    if exiger_complet and (s["reste"] or s["creuses"]):
        log(f"*** {s['reste']} restante(s), {s['creuses']} sans somme : rien a conclure")
        return None
    n = s["n"]
    d = os.path.join(HERE, f"preuve_n{n}_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    os.makedirs(d, exist_ok=True)
    log(f"dossier {d}")

    r = sh([sys.executable, "orchestrator.py", "export"])
    parts = os.path.join(HERE, f"parts_n{n}.txt")
    if not os.path.exists(parts) or os.path.getsize(parts) == 0:
        log("*** export n'a rien ecrit"); return None
    shutil.copy2(parts, d)
    log(f"sommes partielles : {sum(1 for _ in open(parts))} lignes hors SQLite")

    # Instantane par l'API backup de SQLite : une copie de fichier pendant que
    # `run` ecrit donnerait une base dechiree.  Ici c'est transactionnel.
    snap = os.path.join(d, "state.db")
    src = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    dst = sqlite3.connect(snap)
    src.backup(dst)
    # La base source est en WAL, l'instantane herite du mode, et la connexion
    # de destination laisse derriere elle un -wal vide et un -shm.  Ce ne sont
    # pas des pieces du dossier : on les replie dans le fichier principal, sinon
    # l'archive embarque un fichier vide et le controle d'integrite crie a tort.
    dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    dst.execute("PRAGMA journal_mode=DELETE")
    dst.close(); src.close()
    for ext in ("-wal", "-shm"):
        if os.path.exists(snap + ext): os.remove(snap + ext)
    # On relit l'instantane : une sauvegarde qu'on n'a pas rouverte n'est pas
    # une sauvegarde.  C'est le seul exemplaire du travail une fois les
    # machines detruites.
    v = sqlite3.connect(f"file:{snap}?mode=ro", uri=True)
    vu = v.execute("SELECT COUNT(*) FROM tasks WHERE status='done'").fetchone()[0]
    v.close()
    if vu != s["done"]:
        log(f"*** instantane incoherent : {vu} taches faites contre {s['done']} attendues")
        return None
    log(f"instantane de state.db pris et relu ({vu} taches faites)")

    if s["reste"] == 0:
        r = sh([sys.executable, "orchestrator.py", "merge"])
        res = (r.stdout or "") + (r.stderr or "")
        open(os.path.join(d, "resultat.txt"), "w").write(res)
        if r.returncode != 0 or "L(2," not in res:
            log("*** MERGE EN ECHEC -- on ne detruit rien, il faudra refaire des taches")
            log("   ", res.strip()[-400:])
            return None
        for line in res.strip().splitlines(): log("  " + line)
    return d

# ------------------------------------------------- phase 2 : liberation
def teardown():
    if not os.path.exists(INST):
        log("pas de .instances.json : rien a detruire"); return True
    ins = json.load(open(INST)); o = orch()
    for i in ins:
        try:
            r = o.api("DELETE", f"instances/{i['id']}/", {})
            log(f"  instance {i['id']} ({i['n']}x{i['gpu']}) : detruite={r.get('success')}")
        except SystemExit as e:
            log(f"  instance {i['id']} : {str(e)[:140]}")
    vivantes = {x["id"] for x in o.api("GET", "instances/").get("instances", [])}
    encore = [i["id"] for i in ins if i["id"] in vivantes]
    if encore:
        log(f"*** ENCORE ACTIVES : {encore} -- a detruire a la main, le compteur tourne")
        return False
    log("plus aucune instance active : le compteur est arrete")
    os.rename(INST, INST + ".fait")
    return True

# ----------------------------------------------------- phase 3 : preuve
def dossier(d):
    """Audit long et archive. Gratuit : plus rien n'est loue a ce stade."""
    s = state(); n, T = s["n"], s["T"]
    parts = os.path.join(d, f"parts_n{n}.txt")
    log(f"audit (recalcul local d'un echantillon, ~{os.environ.get('SAMPLE','12')} taches)...")
    env = dict(os.environ, SAMPLE=os.environ.get("SAMPLE", "12"))
    r = subprocess.run(["./audit.sh", str(n), str(T), parts], cwd=HERE,
                       capture_output=True, text=True, env=env)
    audit = (r.stdout or "") + (r.stderr or "")
    open(os.path.join(d, "audit.txt"), "w").write(audit)
    audit_ok = r.returncode == 0 and "DIVERGENCE" not in audit
    log("audit :", "OK -- l'echantillon se reproduit au bit pres" if audit_ok
                   else "*** ECHEC, voir audit.txt")

    for f in ("orchestrator.py", "run_batch.sh", "run_shard.sh", "langford6.cu",
              "audit.sh", "verify_all.sh", "dashboard.py", "finalize.py", "README.md"):
        p = os.path.join(HERE, f)
        if os.path.exists(p): shutil.copy2(p, d)

    c = sqlite3.connect(f"file:{os.path.join(d,'state.db')}?mode=ro", uri=True)
    workers = [dict(zip("name kind state ndone spv price dev".split(), row))
               for row in c.execute("SELECT name,kind,state,ndone,spv,price,dev FROM workers")]
    c.close()
    res = open(os.path.join(d, "resultat.txt")).read() if os.path.exists(os.path.join(d, "resultat.txt")) else ""
    json.dump(dict(
        n=n, T=T, faites=s["done"], total=s["total"], complet=(s["reste"] == 0),
        genere_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), audit_ok=audit_ok,
        git_commit=sh(["git", "rev-parse", "HEAD"]).stdout.strip(),
        git_propre=(sh(["git", "status", "--porcelain"]).stdout.strip() == ""),
        source_sha256=sha256(os.path.join(HERE, "langford6.cu")),
        workers=workers,
        instances=json.load(open(INST + ".fait")) if os.path.exists(INST + ".fait")
                  else (json.load(open(INST)) if os.path.exists(INST) else []),
        resultat=[l for l in res.splitlines() if "L(2," in l or "V(n)" in l],
    ), open(os.path.join(d, "manifeste.json"), "w"), indent=1)

    noms = sorted(f for f in os.listdir(d) if f != "SHA256SUMS")
    with open(os.path.join(d, "SHA256SUMS"), "w") as fh:
        for f in noms: fh.write(f"{sha256(os.path.join(d, f))}  {f}\n")
    vides = [f for f in noms if os.path.getsize(os.path.join(d, f)) == 0]
    if vides: log("*** fichiers vides :", vides)

    tgz = d + ".tar.gz"
    with tarfile.open(tgz, "w:gz") as t: t.add(d, arcname=os.path.basename(d))
    with tarfile.open(tgz, "r:gz") as t:
        dedans = {os.path.basename(m.name) for m in t.getmembers() if m.isfile()}
    manque = (set(noms) | {"SHA256SUMS"}) - dedans
    log(f"archive {os.path.basename(tgz)} ({os.path.getsize(tgz)/1024:.0f} Ko)"
        + (f" *** INCOMPLETE : {manque}" if manque else ", relue et complete"))
    return audit_ok and not manque and not vides

def cmd_archive(a):
    d = preserve(exiger_complet=not getattr(a, "partiel", False))
    return dossier(d) if d else False

def cmd_finish(a=None):
    if not cmd_check(): log("on ne detruit rien."); return False
    d = preserve()
    if not d:
        log("*** mise a l'abri incomplete -- ON NE DETRUIT RIEN.")
        log("    Les machines tournent toujours : le resultat passe avant le loyer.")
        return False
    log("resultat sur disque et arithmetiquement valide. Coupure du loyer.")
    coupe = teardown()
    ok = dossier(d)
    log("=" * 60)
    log("resultat  :", "complet et audite" if ok else "SAUVEGARDE, mais audit a revoir")
    log("machines  :", "toutes detruites" if coupe else "*** certaines tournent encore")
    log("dossier   :", d)
    return ok and coupe

def cmd_watch(a):
    """Attend la fin, puis conclut. Ne detruit JAMAIS sur une stagnation : une
    flotte a l'arret veut dire credit epuise ou machines perdues, et c'est le
    moment ou il faut un humain -- pas une suppression automatique."""
    dernier, t0 = -1, time.time()
    while True:
        s = state()
        if s["reste"] == 0 and s["creuses"] == 0:
            log("calcul termine."); return cmd_finish()
        if s["done"] != dernier:
            dernier, t0 = s["done"], time.time()
            log(f"{s['done']}/{s['total']} ({100*s['done']/s['total']:.1f} %), "
                f"{s['reste']} restante(s)")
        elif time.time() - t0 > a.stall * 60:
            log(f"*** aucune tache finie depuis {a.stall} min : flotte a l'arret.")
            log("    Sauvegarde de ce qui est fait, AUCUNE destruction.")
            d = preserve(exiger_complet=False)
            if d: dossier(d)
            return False
        time.sleep(a.every)

if __name__ == "__main__":
    import argparse
    P = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    S = P.add_subparsers(dest="cmd", required=True)
    S.add_parser("check").set_defaults(f=lambda a: cmd_check())
    q = S.add_parser("archive"); q.add_argument("--partiel", action="store_true")
    q.set_defaults(f=cmd_archive)
    S.add_parser("finish").set_defaults(f=lambda a: cmd_finish())
    q = S.add_parser("watch")
    q.add_argument("--every", type=int, default=20, help="intervalle de sondage (s)")
    q.add_argument("--stall", type=int, default=45, help="minutes sans progres = flotte a l'arret")
    q.set_defaults(f=cmd_watch)
    a = P.parse_args()
    sys.exit(0 if a.f(a) is not False else 1)
