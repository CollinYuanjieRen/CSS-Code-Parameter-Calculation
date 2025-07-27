# CSS_Code_Parameter_Calculation
Concise python code to calculate [[n,k,d]].

A couple of **examples** have been put there, like "twoDtori_code.py". 

Simply download and run those examples, which will call the **subroutine** of "CSS_code_parameters.py". 

The **input** of the parameter calculator is imply the two check matrices $H_X$ and $H_Z$ with $\mathbb{F}_2$ coefficients and two switches to calculate the code distance $d_X$ and $d_Z$. 

So it looks like ```compute_css_code_parameters(H_X,H_Z,True,True)```. 

The **output** of ```compute_css_code_parameters``` is "n, r_X (rank of H_X), r_Z (rank of H_Z), k, d_X (distance of H_X), d_Z (distance of H_Z)". 

So $$k=n-r_X-r_Z$$ and $$d$$ is optimized over supports.
