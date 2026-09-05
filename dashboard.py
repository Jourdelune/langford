#!/usr/bin/env python3
"""Tableau de bord de l'avancement, sur http://localhost:8800

    ./dashboard.py [port]

Lit `state.db` en SEULE LECTURE et ne touche a rien : on peut le lancer, le
tuer et le relancer pendant que `run` tourne, sans risque pour le calcul.

Le debit affiche n'est pas celui des `spv` de la table workers -- ce sont des
moyennes glissantes encore ancrees sur leur valeur initiale, donc fausses tant
qu'un worker n'a pas rendu une dizaine de taches.  On le recalcule a partir des
horodatages `utc:` que `run_batch.sh` grave dans la provenance de chaque tache :
c'est une mesure directe, et elle porte sur la chaine complete (SSH, baux,
relances comprises) et non sur le seul noyau CUDA.
"""
import calendar, http.server, json, os, re, socketserver, sqlite3, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DB   = os.path.join(HERE, "state.db")
UTC  = re.compile(r"utc:(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)")

def stamps(rows):
    """Horodatages de fin, en secondes epoch, tries."""
    out = []
    for (prov,) in rows:
        m = UTC.search(prov or "")
        if m:
            # timegm et non mktime : l'horodatage est UTC (run_batch.sh fait
            # `date -u`), or mktime l'interpreterait en heure locale ET y
            # appliquerait la correction d'heure d'ete -- deux heures d'ecart
            # l'ete, assez pour vider la fenetre d'une heure et afficher un
            # debit de zero alors que trente-deux cartes tournent.
            try: out.append(calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")))
            except ValueError: pass
    return sorted(out)

def snapshot():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)
    n = int(c.execute("SELECT v FROM meta WHERE k='n'").fetchone()[0])
    T = int(c.execute("SELECT v FROM meta WHERE k='T'").fetchone()[0])
    st = dict(c.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status"))
    done, tot = st.get("done", 0), sum(st.values())

    ts = stamps(c.execute("SELECT prov FROM tasks WHERE status='done' AND prov IS NOT NULL"))
    now = time.time()
    # Fenetre ADAPTATIVE, et non une heure fixe.  La flotte change en cours de
    # route -- on ajoute des machines, une instance saute, on relance le
    # maitre -- et une fenetre d'une heure melange alors l'ancienne flotte avec
    # la nouvelle : mesure faite le 2026-09-05, 123 taches/h annoncees et 65 h
    # de fin estimee dix-sept minutes apres etre passe de 8 a 32 cartes, contre
    # ~370 taches/h reellement en cours.  On prend donc les vingt dernieres
    # minutes, en se rabattant sur les 40 dernieres taches si c'est trop maigre,
    # et on rapporte le compte au temps ECOULE DEPUIS la plus ancienne d'entre
    # elles -- pas a la largeur nominale de la fenetre, qui surestimerait au
    # demarrage.
    recent = [t for t in ts if t > now - 1200]
    if len(recent) < 20: recent = ts[-40:]
    if len(recent) >= 2:
        rate = len(recent) / max(now - recent[0], 60) * 3600
    else:
        rate = 0.0
    eta = (tot - done) / rate if rate > 0.01 else None

    ws, price = [], 0.0
    for name, kind, state, ndone, spv, p, dev in c.execute(
            "SELECT name,kind,state,ndone,spv,price,dev FROM workers ORDER BY ndone DESC, name"):
        ws.append(dict(name=name, kind=kind, state=state, ndone=ndone or 0,
                       spv=spv, price=p or 0.0, dev=dev))
        if state == "running": price += p or 0.0
    c.close()
    return dict(n=n, T=T, done=done, total=tot, pending=st.get("pending", 0),
                leased=st.get("leased", 0), rate=rate, eta_h=eta,
                cost_h=price, cost_left=(eta * price) if eta else None,
                workers=ws, busy=sum(1 for w in ws if w["state"] == "running"),
                now=time.strftime("%H:%M:%S"))

PAGE = r"""<!-- page servie telle quelle ; aucune dependance externe -->
<title>Langford n=31</title>
<style>
:root{--bg:#f7f7f5;--fg:#1a1a18;--dim:#6b6b66;--line:#dededa;--card:#fff;--ok:#2f7d4f;--warn:#b0641a}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#16161a;--fg:#e8e8e4;--dim:#94948d;--line:#2c2c33;--card:#1e1e24;--ok:#5fbd85;--warn:#e0a259}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:19px;margin:0 0 2px;font-weight:650}
.sub{color:var(--dim);font-size:13px;margin-bottom:22px}
.bar{height:26px;background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden;position:relative}
.fill{height:100%;background:linear-gradient(90deg,#3d7dd8,#5fbd85);transition:width .6s}
.bar b{position:absolute;inset:0;display:grid;place-items:center;font-variant-numeric:tabular-nums;
  font-size:12.5px;text-shadow:0 0 4px var(--bg)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0 26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card .k{color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:21px;font-variant-numeric:tabular-nums;margin-top:3px;font-weight:600}
.card .v small{font-size:12px;font-weight:400;color:var(--dim)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{font-size:11.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em;font-weight:600}
tr:last-child td{border-bottom:0}
.run{color:var(--ok)} .off{color:var(--warn)}
.foot{color:var(--dim);font-size:12px;margin-top:18px}
</style>
<div class=wrap>
  <h1>Langford &mdash; L(2,31)</h1>
  <div class=sub id=sub>chargement...</div>
  <div class=bar><div class=fill id=fill style=width:0></div><b id=pct></b></div>
  <div class=grid id=cards></div>
  <div class=scroll><table><thead><tr><th>worker</th><th>carte</th><th>etat</th>
    <th>faites</th><th>s/vhi</th><th>$/h</th></tr></thead><tbody id=rows></tbody></table></div>
  <div class=foot id=foot></div>
</div>
<script>
const f1=(x,d=1)=>x==null?"&mdash;":x.toFixed(d);
function card(k,v){return `<div class=card><div class=k>${k}</div><div class=v>${v}</div></div>`}
async function tick(){
  let s; try{ s=await (await fetch("/api/status")).json() }catch(e){ return }
  const p = s.total? 100*s.done/s.total : 0;
  document.getElementById("fill").style.width = p.toFixed(2)+"%";
  document.getElementById("pct").innerHTML = `${s.done} / ${s.total} taches &nbsp;&middot;&nbsp; ${p.toFixed(2)} %`;
  document.getElementById("sub").innerHTML =
    `${s.busy} worker(s) actifs &middot; ${s.leased} tache(s) en vol &middot; maj ${s.now}`;
  document.getElementById("cards").innerHTML =
      card("debit", `${f1(s.rate,0)}<small> taches/h</small>`)
    + card("fin estimee", s.eta_h==null?"&mdash;":`${f1(s.eta_h)}<small> h</small>`)
    + card("cout horaire", `${f1(s.cost_h,3)}<small> $/h</small>`)
    + card("reste a payer", s.cost_left==null?"&mdash;":`${f1(s.cost_left,1)}<small> $</small>`);
  document.getElementById("rows").innerHTML = s.workers.map(w=>`<tr>
    <td>${w.name}</td><td>${w.dev>=0?w.dev:"&mdash;"}</td>
    <td class=${w.state=="running"?"run":"off"}>${w.state||"?"}</td>
    <td>${w.ndone}</td><td>${w.spv?w.spv.toFixed(3):"&mdash;"}</td>
    <td>${w.price?w.price.toFixed(3):"&mdash;"}</td></tr>`).join("");
  document.getElementById("foot").textContent =
    "Le debit et la fin estimee viennent des horodatages reels des taches finies "
    + "sur la derniere heure, pas des moyennes glissantes de la base.";
}
tick(); setInterval(tick, 10000);
</script>"""

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass          # pas de bruit dans le terminal
    def do_GET(self):
        if self.path.startswith("/api/status"):
            body, ctype = json.dumps(snapshot()).encode(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8800
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), H) as s:
        print(f"tableau de bord : http://localhost:{port}   (Ctrl-C pour arreter)")
        s.serve_forever()
