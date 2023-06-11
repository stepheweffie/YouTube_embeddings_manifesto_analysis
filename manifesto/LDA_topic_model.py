import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


# Assuming lda_model is your trained LDA model from gensim and corpus and id2word are the corpus and dictionary respectively.
vis_data = gensimvis.prepare(lda_model, corpus, id2word)
pyLDAvis.display(vis_data)
