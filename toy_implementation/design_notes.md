1) Objective & Multi VLM scoring

Aligning with FD, there is a 

- A black-box objective f:X --> R
- For each generated sample x, we compute a score of y = f(x)

we maintain a dataset: D={(x1​,y1​),…,(xn​,yn​)}, Xn​=[x1​,…,xn​],y=[y1​,…,yn​]⊤


VLM pool and per VLM scores
- Let Nv : number of VLMs in the pool (to avoid conflict with FD’s N)
- M = {1,…,Nv​}: indices of VLMs 
- For VLM v ∈ M, its scoring func: sv​(x)∈R (numeric rating that the model v gives for image x for the task)

We sample a subset of VLMs at each evaluation: 

- Let m be the number of VLMs used per evaluation 

For a given eval of x, draw S⊂M,∣S∣=m uniformly with replacement 

Then the vector of raw scores for this evaluation is: 
s(x)={sv​(x):v∈S}.

(z-normalizing per model) - 
sv​(x)= (sv​(x)−μv​​)/σv​+ε

In summary for above, 
1) Sample subset S of VLMs
2) Compute per-VLM scores s(xK)
3) Set y = f(xK) = mean (s(xK))
4) Store both: scalar y in D for FD, vector s(xK) for archive & HT



2) Behaviour Descriptors & Archive 

2.1 - Define a BD function: 
b: X --> R^d
(interim) descriptor dimension (d) = 2, each image x gets a descriptor 
b(x) = (b1​(x),b2​(x))
b1​(x) = Brightness (pixel intensity)
b2(x) = Image entropy 

bj(x) = bj​(x)−minj​​/ (axj​−minj​+ε), j = 1, 2

2.2 - MAP-Elites style archive 

Let: 
nbins (no of bins per dimension) - eg. 5 -> a 5*5 grid 
index set of bins: C ={1,…,nbins​}×{1,…,nbins​}.

2.2 - Normalization
Both BDs are scaled to [0,1] across the current dataset/batch

2.3 - Archive style 

Use a 5*5 MAP elites grid. Each cell stores one elite 
(Image, VD pair & its full score distrbution)

These BDs are lightweight, fast to compute & stable across diverse images. 

3) Replacement rule 
Mann Whitney + effect size. 

For a given cell, let the incumbent elite have score samples

 S_inc = {z1​,…,zn​}
 S_new = {y1​,…,ym​}

 We pool the two sets, rank all values in ascending order and compute the U-stat for candidate group. 

 U = R_new - m(m+1)/2 (R_new is the sum of the ranks of candidate scores)
 If ties are present, the variance of U uses the standard tie-correction term.

Convert U to a z-score (with continuity correction)
 
 z = U−μU​−0.5/ σU and also compute the corresponding p value for the 2 sided test of equal distributions. 

 Additionally, we also need to compute a common-language effect size (CLES)

 CLES = U/mn 

 Replacement decision: 
 Replace incumbent cell iff 

 p < α AND ≥ δ (eg α = 0.05, δ ≈ 0.60)


4) FD integration 

Three core steps are modified from the FD algorithm 

- Score evaluation: It is replaced with multi VLM scoring. Several independent scores are collected for the same image, averaging these scores to produce scalar objective used by FD. 

- Routing into Archive cell: a BD is computed for each image and mapped into a cell. A cell stores at most one elite. 

- Replacement uses Mann-Whitney U instead of sample comparison. 

