import d6tflow
from tasks import OpenAIPDFEmbeddingsTask
import pandas as pd
from visuals.dimensionality_visuals import cosine_similarity_plot, pca_kmeans_dimensionality, tsne_reduce_dimensionality


class EmbeddingsTask(d6tflow.tasks.TaskPqPandas):
    def output(self):
        return d6tflow.targets.PqPandasTarget('data/embeddings_df.parquet')

    def requires(self):
        return OpenAIPDFEmbeddingsTask()

    def run(self):
        data = self.inputLoad()

        print(data)


flow = d6tflow.Workflow()
flow.run(EmbeddingsTask)