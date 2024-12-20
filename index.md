---
layout: single
author_profile: false
toc: true
header:
  image: /assets/images/banner.png
  caption: "Photo by GPT"
  
related: false
---

# Kinase Looking For Love ?

*Elite matchmaking in Switzerland and beyond for select single targets and ligands.*
*For all the **Kinases**, if you feel ready for a new, lasting binding or fast thunderbolt, this tailor-made approach is your best choice.*
*Our team hand-picks your matches and accompanies you throughout the entire process of finding your successful match, from the characterization of your preference to the choice of your soulmate.*

We aim to address the binding problem encountered by a huge amount of kinases in the Kinome. 
Nowadays, despite huge efforts in binding research, finding good matches for kinase remains an open problem.

Our *mantra* : 
 - Kinases the masterpieces: For this very busy and important protein class in the vast Proteom, 
kinases are the ones that contribute the most to human operations (thanks for them) and their dysfunctions underly several diseases. We must take care of them.
 - Overwelmed kinases: Kinases are everywhere and everyone is reaching them, hard to find the good one
 - Match shifting: Kinases encounter problem to keep their match for themself (Off target)

*Matching, binding, and tying with your partner are certainly magic words for us.* \
*But what do we mean by that ?* \
*--> It will depend on **you**, let's see whats binding means for **you**.*

# What is a good match?

*Matching, binding, tying with your partner are certainly magic words for us.* \
*But what do we mean by that?* \
*--> It will depend on **you**!*


<div class="image-container2">
  <button class="image-button"> Whats binding means for you</button>
    <div class="image-box">
      <figure>
        <img src="assets/plots/nan_fraction.png" alt="Plot 1" />
        <figcaption>Binding metric and reaction metric value count </figcaption>
      </figure>
    </div>
</div>

Among all of you we can notice that binding is essentially according to IC50 but we must not ignore the other factors!

> **KI: The big boss** \
>It's the indication of how potent a ligand is on you. It exactly is the quantity of (chemistry)/(alchemy) required to produce half maximum (inhibition)/(fall head over heels). Which is very interesting for us: If we know what makes you fall half in love we will know the perfect factor to make you fall in love completely.

>**IC50: Nice insights** \
>Represent the Half-maximal inhibitory concentration, it's computed in a special situation where the quantity of (chemistry)/(alchemy) is tightly controlled.  It can say a lot but in a very defined situation. It will not give us the objective and timeless match that you need but we need an objective measure that will not depend on the situation. The solution: **KI**

>**IC50 vs KI: bigbro forever** \
Hopefully, both are statistically linked so we can still orient our research by using both of them depending again on you! 

<!--HERE TODO  : reshape the image in the container -->

<div class="image-container2">
  <button class="image-button"> KI vs IC50 Correlation </button>
    <div class="image-box">
      <figure>
        <iframe src="assets/plots/pKi_pIC50.html" width="100%" height="400px"></iframe>
        <figcaption> KI vs IC50 Correlation </figcaption>
      </figure>
  </div>
</div>


> **Temperature and pH rising for your match** \
> Even if the temperature or the pH cannot by themself represent a binding factor, we know that they can influence a lot your match. So let the temperature rise, and trust your pH! 


> So yes, you don't seem to accord a lot of attention to the others, so let's neglect them.

# A Good Candidate 
The success in our matchmaking business depends on how accurately we gather information on our candidates. We want our clients to be able to decide whether it is a hit or a miss in a blink of eye. That is why constructing meaningful profiles is so important. Let's start with the obvious questions – what chemical properties do our clients want in their partners? 

Say that we took them from Rdkit descriptors 

## Categorical features : Fonctional Group
Describe what is a functional group here.

<div class="image-container2">
  <button class="image-button"> Functional Group Categories </button>
    <div class="image-box">
      <figure>
        <img src="assets/plots/functional_groups.png" alt="Plot 1" />
        <figcaption> Functional Group Categories </figcaption>
      </figure>
  </div>
</div>

## Continous features 
Now we consider some continous features and plot the ditrib : 

<!--HERE TODO  reshape the image in the container-->
<div class="image-container2">
  <button class="image-button"> Chemical Features Distribution </button>
    <div class="image-box">
      <figure>
        <img src="assets/plots/chemchar2.png" alt="chemchar" />
        <figcaption> chemical characterization </figcaption>
      </figure>
  </div>
</div>

<div class="image-container2">
  <button class="image-button"> What our members say </button>
    <div class="image-box">
      <figure>
        <iframe src="assets/plots/top_matches.html" width="100%" height="400px"></iframe>
        <figcaption> Top Candidate Ligands characterisation </figcaption>
      </figure>
  </div>
</div>

More complicated that it seems : Reg linear bewteen metric and char 
<div class="image-container2">
  <button class="image-button"> Mutual Info Metric vs Chem Characterization </button>
    <div class="image-box">
      <figure>
        <img src="assets/plots/mutual_information.png" alt="chemchar" />
        <figcaption> Mutual Information bewteen metrcs and chemical characterization </figcaption>
      </figure>
  </div>
</div>


# A match made in Haven  

## Embedding Space From Rdkit

Well, let's go all-in and look for every available piece of information regarding the chemistry of our candidates! Proud of our partnership with RDKit, an undercover chemical detective agency, we chemically profiled the candidates using all Descriptors provided by RDKit. Of course, nobody would want to go through such extensive profiling for the xxx candidates in our catalogues – kinases have better things to do – that's why we propose a conveniently rendered summary obtained through dimensionality reduction techniques, so that our clients could quickly skim through the pages of available ligands and skip a heartbeat when seeing _the one_.

*Note: xxx family was filtered our because its ligands were heavier than the rest of the catalogue*

Oh, this summary doesn't look appealing. We obtained it by keeping the first two components of the PCA of chemical characterization. Maybe that is still too much extra information? Let's go one step further and apply t-SNE on our PCA-reduced profiles.

<div class="plot-container">
  <!-- First Plot -->
  <img src="assets/plots/umap_pIC_RDKIT_descriptors.png" alt="UMAP pIC RDKIT" class="plot active">
  <!-- Second Plot -->
  <img src="assets/plots/umap_pKi_RDKIT_descriptors.png" alt="UMAP pKI RDKIT" class="plot active">
</div>

<div class="switch-buttons">
  <button onclick="switchPlot(0)">IC50</button>
  <button onclick="switchPlot(1)">KI</button>
</div>

<div class="image-container2">
  <button class="image-button"> PCA componants </button>
    <div class="image-box">
      <figure>
        <iframe src="assets/plots/pca_features_RDKIT_descriptors.html" width="100%" height="400px"></iframe>
        <figcaption> PCA componants </figcaption>
      </figure>
  </div>
</div>

Specific to the target ? Familly of protein on this space 

<!--HERE TODO  does not work -->
<div style="text-align: center;">
  <iframe src="assets/plots/umap_target_RDKIT_descriptors.html" class="plot" frameborder="0" style="width: 100%; height: 400px;"></iframe>
  <p><em>RDKIT embedding space color per targets.</em></p>
</div>

Looks better! Now it is clear that our clients definitely have their preferences when it comes to binding partners. Maybe we can go even further and uncover some deeply rooted preferences that even kinases do not know about? Time to dive into machine learning!

## Mol2Vec embedding space

Inspired by our insightful partnership with RDKit, we contacted our next contractor – Mol2Vec. This agency does the dirty work of finding a meaningful representation of chemical properties and similarities between molecules for you. It considers the ligands for what they, in essence, are – atoms connected by bonds – and constructs vectors that capture everything you need to know. Straight to the point, so that our kinases do not lose their precious time on ligands that weren't meant for them from the beginning. Once again, we care about the comfort of our customers, that's way we propose a dimensionally reduced summary of our findings.

<div class="plot-container">
  <!-- First Plot -->
  <img src="assets/plots/umap_pIC_Mol2Vec.png" alt="UMAP pIC Mol2Vec" class="plot active">
  <!-- Second Plot -->
  <img src="assets/plots/umap_pKi_Mol2Vec.png" alt="UMAP pKI Mol2Vec" class="plot active">
</div>

<div class="switch-buttons">
  <button onclick="switchPlot(0)">IC50</button>
  <button onclick="switchPlot(1)">KI</button>
</div>

<div class="image-container2">
  <button class="image-button"> PCA componants </button>
    <div class="image-box">
      <figure>
        <iframe src="assets/plots/pca_features_Mol2Vec.html" width="100%" height="400px"></iframe>
        <figcaption> PCA componants </figcaption>
      </figure>
  </div>
</div>

Specific to the target ? Familly of protein on this space 

<!--HERE TODO  WTF  does not work -->
<div style="text-align: center;">
  <iframe src="assets/plots/umap_target_Mol2Vec.html" class="plot" frameborder="0" style="width: 100%; height: 400px;"></iframe>
  <p><em>Mol2Vec embedding space color per targets</em></p>
</div>

This looks ... convoluted. We contacted Mol2Vec for further explanation but they declined responsibility and accused us of providing ligands that were too structurally similar to be separated. Well, this partnership will not last any longer!


## Morgan Fingerprint embedding space 
<div class="plot-container">
  <!-- First Plot -->
  <img src="assets/plots/umap_pIC_Morgan_Fingerprint.png" alt="UMAP pIC Morgan" class="plot active">
  <!-- Second Plot -->
  <img src="assets/plots/umap_pKi_Morgan_Fingerprint.png" alt="UMAP pKi Morgan" class="plot active">
</div>

<div class="switch-buttons">
  <button onclick="switchPlot(0)">IC50</button>
  <button onclick="switchPlot(1)">KI</button>
</div>


<div class="image-container2">
  <button class="image-button"> PCA componants </button>
    <div class="image-box">
      <figure>
        <iframe src="assets/plots/pca_features_Morgan_Fingerprint.html" width="100%" height="400px"></iframe>
        <figcaption> PCA componants </figcaption>
      </figure>
  </div>
</div>

Specific to the target ? Familly of protein on this space 

<!--HERE TODO  does not work -->
<div style="text-align: center;">
  <iframe src="assets/plots/umap_target_Morgan_Fingerprint.html" class="plot" frameborder="0" style="width: 100%; height: 400px;"></iframe>
  <p><em>Morgan Fingerprint embedding space color per targets</em></p>
</div>


## Our very own embedding

Luckily, our agency has an R&D department of its own. Who needs rude contractors, anyway? After long and tedious development, we present our proprietary machine-learning method for ad hoc embeddings, because our clients deserve the best. We first trained a transformer that predicted a ligand's SMILES provided the beginning of its SMILES (a _Not-so-large SMILES model_ if you wish). The trick is that the model, in fact, generated a vector of probabilities of the possible SMILES characters – that is our embedding! Ingenious, isn't it?

Let's look at our Big Matchmaker.

<div class="plot-container">
  <!-- First Plot -->
  <img src="assets/plots/umap_pIC_full.png" alt="UMAP pIC all" class="plot active">
  <!-- Second Plot -->
  <img  src="assets/plots/umap_pKi_full.png" alt="UMAP pIC all" class="plot active">
</div>

<div class="switch-buttons">
  <button onclick="switchPlot(0)">IC50</button>
  <button onclick="switchPlot(1)">KI</button>
</div>

<div class="image-container2">
  <button class="image-button"> PCA componants </button>
    <div class="image-box">
      <figure>
        <iframe src="assets/plots/pca_features_full.html" width="100%" height="400px"></iframe>
        <figcaption> PCA componants </figcaption>
      </figure>
  </div>
</div>

Specific to the target ? Familly of protein on this space 

<!--HERE TODO  does not work -->
<div style="text-align: center;">
  <iframe src="assets/plots/umap_target_full.html" class="plot" frameborder="0" style="width: 100%; height: 400px;"></iframe>
  <p><em>Full embedding space color per targets</em></p>
</div>


# Special space for special guests

As it turns out, being a matchmaking agency is no joke. We found our disappointed client listenting to _You Should Be Stronger Than Me_ by Amy Winehouse on repeat.

