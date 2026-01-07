from neuralnet.dqn_trainer import DQNTrainer


def main():
    trainer = DQNTrainer(load_checkpoint="best_model_freeway.pth")
    trainer.train()

if __name__ == "__main__":
    main()