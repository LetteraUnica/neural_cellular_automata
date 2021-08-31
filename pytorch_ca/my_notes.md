## Caso in cui la perturbazione è globale    
• [ ] Aumentare il parametro $\lambda$ di regolarizzazione  
• [x] Applicare la pertubazione in una botta dopo un certo numero di steps  
• [ ] Allenare la pertubazione 

## Virus (perturbazione locale)  
Nel caso in cui si fa con un virus bisogna vedere come cambiare la maschera in un modo che sia realistico  

uno modo sarebbe il seguente:  
• La maschera della regola $i$-esima è definita come quel posto dove le $\alpha_i>0.1$  
• Se $\alpha_i\ge 0.1$ allora $\alpha_j=0\quad \forall j\neq i$  
• Bisogna vedere che succede quando tutti gli $\alpha_i<0.1$ , le possibilità sono che:  
• [ ] Si lascia fare a tutte le regole del CA assieme e chi prima arriva a 0.1 vince  
• [ ] Si usa un metodo stocastico per la scelta  
• [ ] Studiare vari tipi di condizioni iniziali di aggiunta di pertubazioni