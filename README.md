# CSS-Code-Parameter-Calculation
Concise python code to calculate [[n,k,d]].

A couple of **examples**, including "twoDtori_code.py" the 2D toric code, have been provided. 

Simply download and run those examples, which will call the **subroutine** of "CSS_code_parameters.py". 

The **input** of the parameter calculator is imply the two check matrices $H_X$ and $H_Z$ with $\mathbb{F}_2$ coefficients and two switches ON/OFF determining if we calculate the code distance $d_X$ and $d_Z$ or not. 

So it looks like ```compute_css_code_parameters(H_X,H_Z,True,True)```. 

The **output** of ```compute_css_code_parameters``` is "n, r_X (rank of H_X), r_Z (rank of H_Z), k, d_X (distance of H_X), d_Z (distance of H_Z)". 

So $$k=n-r_X-r_Z$$ and $$d$$ is optimized over supports. 

Let me know if there's any bug or error.
