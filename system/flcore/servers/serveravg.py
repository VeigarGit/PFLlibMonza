import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
    def normalize_entropies(self, client_entropies):
        """Normaliza as entropias para que fiquem no intervalo [0, 1]"""
        # Obter as entropias
        entropies = np.array(list(client_entropies.values()))

        # Calcular o valor mínimo e máximo
        min_entropy = np.min(entropies)
        max_entropy = np.max(entropies)

        # Normalizar as entropias
        normalized_entropies = (entropies - min_entropy) / (max_entropy - min_entropy)

        # Atualizar o dicionário com as entropias normalizadas
        normalized_client_entropies = {client_id: normalized_entropy for client_id, normalized_entropy in zip(client_entropies.keys(), normalized_entropies)}

        # Exibir as entropias normalizadas
        for client_id, normalized_entropy in normalized_client_entropies.items():
            print(f"Normalized Shannon entropy for client {client_id}: {normalized_entropy:.4f}")

        return normalized_client_entropies

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if i>0:
                if self.cc==0:
                    global_model_params = list(self.global_model.parameters()) 
                # Calcular a similaridade de cosseno entre os modelos dos clientes e o modelo global
                    similarities = self.calculate_similarity_with_global_model(global_model_params)
                    for sim in similarities:
                        print(f"Cosine similarity between client {sim[0]} and the global model: {sim[1]:.4f}")
                if self.cc==1:
                    similarity_scores = self.calculate_similarity_scores()
                    for client_id, score in similarity_scores.items():
                        print(f"Cosine similarity for client {client_id}: {score:.4f}")
                    normalized_client_entropies = self.normalize_entropies(similarity_scores)
                if self.cc ==2:
                    client_entropies = self.calculate_client_entropies()
                    normalized_client_entropies = self.normalize_entropies(client_entropies)
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
