# import os
# import pickle


# class RobertaHumor:
#     """Class for working with Roberta models trained on HRI Datasets"""

#     """
#     This is a class for working with models trained on one of the joke data sets.
#     Supported datasets:

#     Pipelines for training can be found in the organization on GitHub:
#     https://github.com/Humor-Research-Institute/
#     """

#     def __init__(self, name) -> None:
#         self.supported_models = []
#         if name not in self.supported_models:
#             raise ValueError("The model is not yet trained on this data! Check supported datasets.")
        
#         user_home = os.getenv('HOME')
#         self.hri_roberta_path = os.path.join(user_home, 'hri_tools_data/', 'models/', 'RoBerta/')
#         if not os.path.exists(self.hri_roberta_path):
#             raise ValueError("Models path is unreachable! Check your local files.")

#     def load(self):
#         pass

#     def predict(self):
#         pass  


# class NaiveBayesHumor:
#     """Class for working with Naive Bayes models (scikit-learn) trained on HRI Datasets"""

#     """
#     This is a class for working with models trained on one of the joke data sets.
#     Supported datasets:

#     Pipelines for training can be found in the organization on GitHub:
#     https://github.com/Humor-Research-Institute/
#     """

#     def __init__(self, name) -> None:
#         self.supported_models = []
#         if name not in self.supported_models:
#             raise ValueError("The model is not yet trained on this data! Check supported datasets.")
        
#         user_home = os.getenv('HOME')
#         self.hri_roberta_path = os.path.join(user_home, 'hri_tools_data/', 'models/', 'NaiveBayes/')
#         if not os.path.exists(self.hri_roberta_path):
#             raise ValueError("Models path is unreachable! Check your local files.")

#     def load(self):
#         pass

#     def predict(self):
#         pass
