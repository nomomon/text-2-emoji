from text2emoji.models.eval_model import eval_best_model

if __name__ == '__main__':
    # expects the API to be running

    eval_best_model("word2vec", "test")
    eval_best_model("unfrozen_bert", "test")
    eval_best_model("mobert", "test")
