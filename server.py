from flask import Flask
from view.PageView import PageView
from view.CalcView import CalcView

app = Flask(__name__)
PageView.register(app)
CalcView.register(app)

@app.route("/clone_master",methods=["POST"])
def clone_master():
    """Il metodo crea un endpoint da fornire a git hub affinchè possa gestire in autonomia le pull in arrivo sul branch
    di nostro interesse. I dati di configurazione sono gestiti all'interno del file Config/Git/PAT.json"""
    try:
        # salvo il nome delle variabili che servono per scaricare gli aggiornamenti
        branch_name, repo_name = "master", "StatLog"
        repo = git.Repo(f"./")
        origin = repo.remotes.origin
        repo.create_head(f'{branch_name}',
                         origin.refs.Testing).set_tracking_branch(origin.refs.Testing).checkout()
        origin.pull()
        # conclusa la sincronizzazione con git hub ritorno un success
        return f"Sincronizzazione conclusa con successo", 200
    except Exception as e:
        return f"Errore in fase di sincronizzazione con Git:<br>{e}", 404
