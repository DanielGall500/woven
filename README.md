# Word-Order Variation Encoder (WOVEN)
## Improving L2 Structure Comprehension with SHAP-based Explainability

#### Overview
Language structure can be a major obstacle for learners in acquiring a second language (L2). This project strives to remedy some of these difficulties by detecting and expressing the variable word order in English-German translations in an intuitive and helpful way. In the final build, the application should take a sentence in the source language English, translate it into the target language German, analyse the translation for the variation in input-output word order formation, and lastly express this to the user in a way which will help them understand the differing structure.

A Neural Machine Translation (NMT) model is the first step in the development cycle. Marian is an efficient and open-source NMT framework written in pure C++ and with minimal dependencies. Hugging Face provides a pre-trained Marian MT model which can be used and deployed in Python. At this layer, the focus is solely on translation accuracy.

The second development phase will relate to the explanation layer. **SHAP** is a model-agnostic explainability framework that will be responsible for drawing the relations between input tokens and output tokens.

Here the focus shifts towards *explainability* or *interpretability*. That is, why did the model choose to make the structure of the translation so, and answering the question of which input elements were “responsible” for individual output elements (i.e which word in English corresponds to which in German). Note that this is inherently different from searching an English-German dictionary, as the explainer model is "aware" of which English elements are related to which German elements without any prerequisite knowledge of English or German and can account for the issues that arise in attempting to directly translate between the two. The word order variations between each language will be found and stored by WOVEN (Word-Order Variation Encoder), which will then be used in the next phase to communicate these variations to the learner.

The final phase develops a web-based application running on Flask. Flask is a web application micro-framework which sacrifices built-in functionality for a simpler code base and reduced overhead. It is suitable for the scope of this project, adhering to the KISS principle of project design. This end of this phase marks the final of three in the project's development.

#### NMT Model BLEU Score
The WOVEN NMT model achieved a BLEU score of 31. This falls into the range of ‘understandable to good translations’, and achieves what I set out to do in this phase of the project: create a model which achieves reasonable but not necessarily great translation accuracy using a well-defined metric. Note that this score would likely be somewhat higher if multiple reference sentences were used for each single source sentence, however in this case we limited our score in that respect in order to fall in line with WMT20 standards. The n-gram precision scores can be seen below.
