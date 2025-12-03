class Pipeline:
    def __init__(self, name, model_settings:dict, pollution_settings:dict):
        self.name = name
        self.model_settings = model_settings
        self.pollution_settings = pollution_settings

    def run(self, dataset_name):
        pass