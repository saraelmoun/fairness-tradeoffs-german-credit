  <div align="center">                                                          
                                                                                
  # Fairness Trade-offs on German Credit                                     
                                                                                
  <img src="notebooks/JUGE.jpg" width="280"/>    

<div align="center" style="             
font-family: Georgia, serif;                                                
background-color: #782038;                                                  
padding: 1.3em 2em;       
margin: 1.5em auto;                                                         
max-width: 550px;                                                        
">                                      
<p style="color:#ffffff; font-size:1.2em; margin:0;">
  <b>Elamine Mohammed</b> &nbsp;&amp;&nbsp; <b>Sara El Mountasser</b>       
</p>                                                                 
<p style="color:#e8a8bc; font-style:italic; font-size:0.95em; margin:0.5em 0
0 0;">                                                                       
  sous la direction de Charlotte Laclau                                     
</p>                                   
</div>                                                                        
                                                                                
  ### *Are we fair?* 

❯ Comment évaluer l'équité d'un modèle de scoring crédit envers les jeunes        
emprunteurs, et jusqu'où les techniques classiques de mitigation                
permettent-elles de la rétablir ?  

---
                                                                                
1. Comment concilier équité algorithmique et viabilité économique dans un       
système de scoring crédit ?                                                     
                                                                                
2. Quelles sont les limites des techniques classiques de mitigation des biais,  
et d'où proviennent ces limites ?                                               
                                                                                
3. Que nous apprennent les outils d'interprétation sur la nature réelle des     
biais observés ?                                                                

</div>

---



## Structure

1. **[NoteD'analyse.pdf](NoteD'analyse.pdf)** — Synthèse.
2. **[notebooks/IADATA708_GermanCredit.ipynb](notebooks/IADATA708_GermanCredit.ipynb)** - Notebook Principal . 
3. **[notebooks/ExplicationHardt.ipynb](notebooks/ExplicationHardt.ipynb)** — Annexe pédagogique. 
4. **[contrefactual-recourse/](contrefactual-recourse/)** — *bonus* interprétabilité. Voir le
   [README dédié](contrefactual-recourse/README.md).

## Installation

Python 3.10+ recommandé.

```bash
python -m venv .venv
source .venv/bin/activate

# notebooks principaux
pip install -r requirements.txt

# bonus PACR-AP (à installer en plus si besoin)
pip install -r contrefactual-recourse/requirements.txt
```

