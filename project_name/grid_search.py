from models.grid_search_model import GridSearchModel

if __name__ == "__main__":
    grid_search_model = GridSearchModel()
    grid_search_model.run()
    print(grid_search_model.get_best_hyperparameters())
