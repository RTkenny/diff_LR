# **Finetuning diffusion with recursive likelihood ratio method**

for icml25 

Text2Image folder for the experiment of SD1.4
Text2Video folder for the experiment of videocrafter
## To do
- [x] modified the wandb logging 
- [x] reorganize the diffused chain
- [x] implement zero-order estimator
- [x] implement the Recursive Likelihood Ratio estimator
- [ ] implement low rank estimator

# advantage and intuition of Diff_LR
- no good value function is available
- memory efficient
- low rank structure adaption