# YouTube_embeddings_manifesto_analysis
This is a set of counterterrorism machine learning and data processing tools 
for NLP analysis of a terrorist's manifesto against transcripts from YouTube videos by channel, playlist, or individual video ID. 

It assumes unsupervised learning from scratch and begins basic text wrangling for categories and clusters. 

There are two sets of embeddings; text-embeddings-ada-002 from OpenAI and BERT from Hugging Face Transformers. They are generally used for different applications, but are combined for this use case. 

The ada-002 embeddings takes a hyperparameter for KMeans clusters based on semantic sentiment similarity. This technique is used for grouping similar text by several authors, i.e. in app ratings for a retailer's POS, a forum, a stream of posts, etc

This is a comparison to a lethal terrorist's manifesto. His document is a well known example of plagiarism and copy/paste consipiracy theories. 

This project contains the transcripts for the @JordanBPeterson Youtube

The single video being compared in the manifesto directory is Jordan's "Who Is Teaching Your Kids?" from PragerU.

