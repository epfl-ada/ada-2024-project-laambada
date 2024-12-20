---
layout: single
author_profile: false
toc: true
header:
  image: /assets/images/banner.png
  caption: "Photo by GPT"
  
related: false
---

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
        <iframe src="/assets/plots/nan_fraction.html" width="100%" height="400px"></iframe>
        <figcaption> Metric Values counts </figcaption>
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

<div class="image-container2">
  <button class="image-button"> KI vs IC50 Correlation </button>
    <div class="image-box">
      <figure>
        <iframe src="/assets/plots/pKi_pIC50.html" width="100%" height="400px"></iframe>
        <figcaption> KI vs IC50 Correlation </figcaption>
      </figure>
  </div>
</div>


> **Temperature and pH rising for your match** \
> Even if the temperature or the pH cannot by themself represent a binding factor, we know that they can influence a lot your match. So let the temperature rise, and trust your pH! 


> So yes, you don't seem to accord a lot of attention to the others, so let's neglect them.

# A Good Candidate 

Rdkit checical feature extraction !

<div class="image-container2">
  <button class="image-button"> Functional Group Categories </button>
    <div class="image-box">
      <figure>
        <iframe src="/assets/plots/functional_groups.html" width="100%" height="400px"></iframe>
        <figcaption> Functional Group Categories </figcaption>
      </figure>
  </div>
</div>

<div class="image-container2">
  <button class="image-button"> chemchar </button>
    <div class="image-box">
      <figure>
        <iframe src="/assets/plots/chemchar2.html" width="100%" height="400px"></iframe>
        <figcaption> chem char </figcaption>
      </figure>
  </div>
</div>

# A match made in Haven 

The success in our matchmaking business depends on how accurately we gather information on our candidates. We want our clients to be able to decide whether it is a hit or a miss in a blink of eye. That is why constructing meaningful profiles is so important. Let's start with the obvious questions – what chemical properties do our clients want in their partners?  

## Embedding Space From Rdkit

Well, let's go all-in and look for every available piece of information regarding the chemistry of our candidates! Proud of our partnership with RDKit, an undercover chemical detective agency, we chemically profiled the candidates using all Descriptors provided by RDKit. Of course, nobody would want to go through such extensive profiling for the xxx candidates in our catalogues – kinases have better things to do – that's why we propose a conveniently rendered summary obtained through dimensionality reduction techniques, so that our clients could quickly skim through the pages of available ligands and skip a heartbeat when seeing _the one_.

*Note: xxx family was filtered our because its ligands were heavier than the rest of the catalogue*

Oh, this summary doesn't look appealing. We obtained it by keeping the first two components of the PCA of chemical characterization. Maybe that is still too much extra information? Let's go one step further and apply t-SNE on our PCA-reduced profiles.


<iframe src="/assets/plots/reduction_plots_RDKIT_descriptors.html" width="100%" height="400px"></iframe>

<div class="image-container2">
  <button class="image-button"> PCA componants </button>
    <div class="image-box">
      <figure>
        <iframe src="/assets/plots/pca_features_RDKIT_descriptors.html" width="100%" height="400px"></iframe>
        <figcaption> PCA componants </figcaption>
      </figure>
  </div>
</div>

Looks better! Now it is clear that our clients definitely have their preferences when it comes to binding partners. Maybe we can go even further and uncover some deeply rooted preferences that even kinases do not know about? Time to dive into machine learning!

## Mol2Vec embedding space

Inspired by our insightful partnership with RDKit, we contacted our next contractor – Mol2Vec. This agency does the dirty work of finding a meaningful representation of chemical properties and similarities between molecules for you. It considers the ligands for what they, in essence, are – atoms connected by bonds – and constructs vectors that capture everything you need to know. Straight to the point, so that our kinases do not lose their precious time on ligands that weren't meant for them from the beginning. Once again, we care about the comfort of our customers, that's way we propose a dimensionally reduced summary of our findings.

<iframe src="/assets/plots/reduction_plots_Mol2Vec.html" width="100%" height="400px"></iframe>

<div class="image-container2">
  <button class="image-button"> PCA componants </button>
    <div class="image-box">
      <figure>
        <iframe src="/assets/plots/pca_features_Mol2Vec.html" width="100%" height="400px"></iframe>
        <figcaption> PCA componants </figcaption>
      </figure>
  </div>
</div>

This looks ... convoluted. We contacted Mol2Vec for further explanation but they declined responsibility and accused us of providing ligands that were too structurally similar to be separated. Well, this partnership will not last any longer!


## Morgan Fingerprint embedding space 


<iframe src="/assets/plots/reduction_plots_Morgan_Fingerprint.html" width="100%" height="400px"></iframe> 

<div class="image-container2">
  <button class="image-button"> PCA componants </button>
    <div class="image-box">
      <figure>
        <iframe src="/assets/plots/pca_features_Morgan_Fingerprint.html" width="100%" height="400px"></iframe>
        <figcaption> PCA componants </figcaption>
      </figure>
  </div>
</div>

## Our very own embedding

Luckily, our agency has an R&D department of its own. Who needs rude contractors, anyway? After long and tedious development, we present our proprietary machine-learning method for ad hoc embeddings, because our clients deserve the best. We first trained a transformer that predicted a ligand's SMILES provided the beginning of its SMILES (a _Not-so-large SMILES model_ if you wish). The trick is that the model, in fact, generated a vector of probabilities of the possible SMILES characters – that is our embedding! Ingenious, isn't it?

Let's look at our Big Matchmaker.

<iframe src="/assets/plots/reduction_plots_full.html" width="100%" height="400px"></iframe> 

<div class="image-container2">
  <button class="image-button"> PCA componants </button>
    <div class="image-box">
      <figure>
        <iframe src="/assets/plots/pca_features_full.html" width="100%" height="400px"></iframe>
        <figcaption> PCA componants </figcaption>
      </figure>
  </div>
</div>

# Special space for special guests

As it turns out, being a matchmaking agency is no joke. We found our disappointed client listenting to _You Should Be Stronger Than Me_ by Amy Winehouse on repeat.

