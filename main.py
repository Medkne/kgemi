from trainer import Trainer
from utils import Dataloader, accuracy
import argparse
import torch


def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=1000, type=int, help="number of epochs")
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.001, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="MKG", type=str, help="wordnet dataset")
    parser.add_argument('-emb_dim', default=500, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-batch_size', default=256, type=int, help="batch size")
    parser.add_argument('-save_each', default=50, type=int, help="validate every k epochs")
    parser.add_argument('-model', default='SimplE', type=str, help="Model = [TransE, SImplE, DistMult]")
    parser.add_argument('-margin', default=1., type=float, help="margin for pairwise loss")
    parser.add_argument('-loss', default= 'Pointwise', type=str, help="Loss = [Pairwise, Pointwise]")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    data = Dataloader('datasets' + "/" + args.dataset)
    train = data.load('train')
    valid = data.load('valid')
    test = data.load('test')
    num_ent = data.num_ent()
    num_rel = data.num_rel()
    ent2id = data.ent2id
    rel2id = data.rel2id

    print("~~~~ Training ~~~~")
    trainer = Trainer(train, num_ent, num_rel, ent2id, rel2id ,args)
    trainer.train()
    print("~~~~ Select best epoch on validation set ~~~~")
    epochs2test = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]    
    best_accuracy = 0.0
    best_epoch = "0"
    for epoch in epochs2test:
        print(epoch)
        model_path = "models/" + 'MKG' + "/" + epoch + ".chkpnt"
        model = torch.load(model_path, map_location = trainer.device)
        model.eval()
        acc, f1, class_labels = accuracy(valid, ent2id, rel2id, model)
        if acc > best_accuracy:
            best_accuracy = acc
            best_epoch = epoch
    print("Best epoch: " + best_epoch)
    print("---- Test on the best epoch ----")
    best_model_path = "models/" + "MKG" + "/" + best_epoch + ".chkpnt"
    best_model = torch.load(best_model_path, map_location = trainer.device)
    best_model.eval()
    acc, fscores, class_labels = accuracy(test, ent2id, rel2id, best_model)
    print(f"Accuracy: {round(acc, 2)}")
    print(f"F1-score: {[round(f1, 2) for f1 in fscores]}")
    print("classes:", class_labels)