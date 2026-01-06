import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import awgn_channel, reparameterize, gaussian_kl_divergence




class CommonNetwork(nn.Module):
    def __init__(self, n_in, n_c_latent, n_c_h):
        super(CommonNetwork, self).__init__()

        self.n_in = n_in
        self.n_c_latent = n_c_latent
        self.n_h = n_c_h

#------ CU-Encoder
        self.cu_enc = nn.Sequential(
            nn.Linear(n_in, n_c_h), nn.ReLU(),
            nn.Linear(n_c_h, n_c_h), nn.ReLU(),
            
        )
        self.cu_enc_mu = nn.Linear(n_c_h, n_c_latent)
        self.cu_enc_ln_var = nn.Linear(n_c_h, n_c_latent)

#------ CU-Decoder (auxiliary)
        self.cu_dec_first = nn.Sequential(
            nn.Linear(n_c_latent,n_c_latent), nn.ReLU(),
        )
        self.cu_dec_first_out = nn.Linear(n_c_latent, 1)

        self.cu_dec_second = nn.Sequential(
            nn.Linear(n_c_latent,n_c_latent), nn.ReLU(),
        )
        self.cu_dec_second_out = nn.Linear(n_c_latent, 10)

    def cu_encode(self, s):
        h = self.cu_enc(s)
        return self.cu_enc_mu(h), self.cu_enc_ln_var(h)
    def cu_decode_first(self, c):
        h = self.cu_dec_first(c)
        return self.cu_dec_first_out(h)
    def cu_decode_second(self, c):
        h = self.cu_dec_second(c)
        return self.cu_dec_second_out(h)
    


    
class SpecificNetwork_first(nn.Module):
    def __init__(self, n_x_in, n_x_latent, n_x_h):
        super(SpecificNetwork_first, self).__init__()

        self.n_x_in = n_x_in
        self.n_x_latent = n_x_latent
        self.n_x_h = n_x_h

#------ SU1-Encoder
        self.su_enc_first = nn.Sequential(
            nn.Linear(n_x_in, n_x_h), nn.ReLU(),
            nn.Linear(n_x_h, n_x_h), nn.ReLU(),
        )
        self.su_enc_first_mu = nn.Sequential(
            nn.Linear(n_x_h, n_x_latent), nn.Tanh()
        )
        self.su_enc_first_ln_var = nn.Sequential(
            nn.Linear(n_x_h, n_x_latent), nn.Sigmoid()
        )

        # SU1-Decoder
        self.su_dec_first = nn.Sequential(
            nn.Linear(n_x_latent, n_x_latent), nn.ReLU(),
        )
        self.su_dec_first_out = nn.Linear(32, 1)

    def su_encode_first(self, c):
        h = self.su_enc_first(c)
        return self.su_enc_first_mu(h), self.su_enc_first_ln_var(h)

    def su_decode_first(self, x):
        h = self.su_dec_first(x)
        return self.su_dec_first_out(h)
    



class SpecificNetwork_second(nn.Module):
    def __init__(self, n_x_in, n_x_latent, n_x_h):
        super(SpecificNetwork_second, self).__init__()

        self.n_x_in = n_x_in
        self.n_x_latent = n_x_latent
        self.n_x_h = n_x_h

#------ SU2-Encoder
        self.su_enc_second = nn.Sequential(
            nn.Linear(n_x_in, n_x_h), nn.ReLU(),
            nn.Linear(n_x_h, n_x_h), nn.ReLU(),
        )
        self.su_enc_second_mu =nn.Sequential(
            nn.Linear(n_x_h, n_x_latent), nn.Tanh()
        )
        self.su_enc_second_ln_var = nn.Sequential(
            nn.Linear(n_x_h, n_x_latent), nn.Sigmoid()
        )

        # SU2-Decoder
        self.su_dec_second = nn.Sequential(
            nn.Linear(n_x_latent, n_x_latent), nn.ReLU(),
        )
        self.su_dec_second_out = nn.Linear(32, 10) 

    
    def su_encode_second(self, c):
        h = self.su_enc_second(c)
        return self.su_enc_second_mu(h), self.su_enc_second_ln_var(h)

    def su_decode_second(self, x):
        h = self.su_dec_second(x)
        return self.su_dec_second_out(h)
    


class SpecificNetwork_ohneCU(nn.Module):
    def __init__(self, n_x_latent, n_x_h):
        super(SpecificNetwork_ohneCU, self).__init__()

        self.n_x_latent = n_x_latent
        self.n_x_h = n_x_h

#------ SU-ohneCU-Encoder
        self.su_enc_ohnecu = nn.Sequential(
            nn.Linear(784, n_x_h), nn.ReLU(),
            nn.Linear(n_x_h, n_x_h), nn.ReLU(),
        )
        self.su_enc_ohnecu_mu =nn.Sequential(
            nn.Linear(n_x_h, n_x_latent), nn.Tanh()
        )
        self.su_enc_ohnecu_ln_var = nn.Sequential(
            nn.Linear(n_x_h, n_x_latent), nn.Sigmoid()
        )

        # SU-ohneCU-Decoder
        self.su_dec_ohnecu = nn.Sequential(
            nn.Linear(n_x_latent, n_x_latent), nn.ReLU(),
        )
        self.su_dec_ohnecu_out = nn.Linear(32, 1) 

    
    def su_encode_ohnecu(self, c):
        h = self.su_enc_ohnecu(c)
        return self.su_enc_ohnecu_mu(h), self.su_enc_ohnecu_ln_var(h)

    def su_decode_ohnecu(self, x):
        h = self.su_dec_ohnecu(x)
        return self.su_dec_ohnecu_out(h)




class MultiTaskMultiUserComm:

    def __init__(self, n_in, n_c_latent, n_c_h, n_x_latent, n_x_h):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.common_network = CommonNetwork(n_in=n_in, n_c_latent= n_c_latent, n_c_h=n_c_h).to(self.device)
        self.specific_network_first = SpecificNetwork_first(n_x_in=n_c_latent, n_x_latent=n_x_latent, n_x_h=n_x_h).to(self.device)
        self.specific_network_second = SpecificNetwork_second(n_x_in=n_c_latent, n_x_latent=n_x_latent, n_x_h=n_x_h).to(self.device)
        self.specific_network_ohnecu = SpecificNetwork_ohneCU(n_x_latent=n_x_latent, n_x_h=n_x_h).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.categ_criterion = torch.nn.CrossEntropyLoss() 
        

        self.train_loss_cu = []
        self.train_loss_su1 = []
        self.train_loss_su2 = []
        self.train_loss_su_ohnecu = []
        self.accuracy_task_1 = []
        self.accuracy_task_2 = []
        self.accuracy_task_ohnecu = []
        self.error_task_1 = []
        self.error_task_2 = []
        self.error_task_ohnecu = []

        self.train_time = []
              

    def _loss_CU_task1(self, s, targets1):
        c_mu, c_ln_var = self.common_network.cu_encode(s)
        c = reparameterize(c_mu, c_ln_var)
        z = self.common_network.cu_decode_first(c)
        targets1 = targets1.float()
        re = self.criterion(z, targets1.view(-1,1))
        re_sum = torch.sum(re)
        loss_cu1 = re_sum
        return loss_cu1
    
    def _loss_CU_task2(self, s, targets2):
        c_mu, c_ln_var = self.common_network.cu_encode(s)
        c = reparameterize(c_mu, c_ln_var)
        z = self.common_network.cu_decode_second(c)
        re = self.categ_criterion(z, targets2)
        re_sum = torch.sum(re)
        loss_cu2 = re_sum
        return loss_cu2



    def _loss_SU_first(self, s, targets1):
        c_mu, c_ln_var = self.common_network.cu_encode(s)
        c = reparameterize(c_mu, c_ln_var)
        x_mu, x_ln_var = self.specific_network_first.su_encode_first(c)
        x = reparameterize(x_mu, x_ln_var)
        x_hat = awgn_channel(x)
        z     = self.specific_network_first.su_decode_first(x_hat)
        targets1 = targets1.float()
        sre = self.criterion(z, targets1.view(-1,1))  # This takes care of sigmoid itself and has a negative
        kl = gaussian_kl_divergence(x_mu, x_ln_var)
        sre_sum = torch.sum(sre) # sre stands for semantic recovery error
        kl_sum = torch.sum(kl)
        loss_su1 = sre_sum + 0.001*kl_sum
        return loss_su1


    def _loss_SU_second(self, s, targets2):
        c_mu, c_ln_var = self.common_network.cu_encode(s)
        c = reparameterize(c_mu, c_ln_var)
        x2_mu, x2_ln_var = self.specific_network_second.su_encode_second(c)
        x2 = reparameterize(x2_mu, x2_ln_var)
        x2_hat = awgn_channel(x2)
        z2 = self.specific_network_second.su_decode_second(x2_hat)
        sre2 = self.categ_criterion(z2, targets2) 
        kl = gaussian_kl_divergence(x2_mu, x2_ln_var)
        sre2_sum = torch.sum(sre2)
        kl_sum = torch.sum(kl)
        loss_su2 = sre2_sum + 0.001*kl_sum
        return loss_su2
    
    
    def _loss_SU_ohnecu(self, s, targets1):
        x_mu, x_ln_var = self.specific_network_ohnecu.su_encode_ohnecu(s)
        x = reparameterize(x_mu, x_ln_var)
        x_hat = awgn_channel(x)
        z     = self.specific_network_ohnecu.su_decode_ohnecu(x_hat)
        targets1 = targets1.float()
        sre = self.criterion(z, targets1.view(-1,1))  # This takes care of sigmoid itself and has a negative
        kl = gaussian_kl_divergence(x_mu, x_ln_var)
        sre_sum = torch.sum(sre) # sre stands for semantic recovery error
        kl_sum = torch.sum(kl)
        loss_su_ohne = sre_sum + 0.001*kl_sum
        return loss_su_ohne

    

    def fit(self, first_dataset, second_dataset, first_test_dataset, second_test_dataset, batch_size=100,
            n_epoch_primal=500, learning_rate=0.01, path=None):

        
        N = 100                                         

        first_loader = DataLoader(first_dataset, batch_size=batch_size, shuffle=True)
        second_loader = DataLoader(second_dataset, batch_size=batch_size, shuffle=True)
     

        optimizer_cu = torch.optim.Adam(self.common_network.parameters(), lr=learning_rate)
        optimizer_su_first = torch.optim.SGD(self.specific_network_first.parameters(), lr= 0.01) # for SUs we need to use SGD to ignore the DRE networks in updates
        optimizer_su_second = torch.optim.SGD(self.specific_network_second.parameters(), lr= 0.01)
        optimizer_su_ohnecu = torch.optim.SGD(self.specific_network_ohnecu.parameters(), lr= 0.01)

        for epoch_primal in range(n_epoch_primal):
            start = time.time()


            mean_loss_cu = 0
            mean_loss_cu1 = 0
            mean_loss_cu2 = 0
            
            mean_loss_su1 = 0
            mean_loss_su2 = 0

            mean_loss_su_ohnecu = 0

            self.specific_network_first.train()


            zipped_dataloader = zip(first_loader, second_loader)
            
            # Training CU
            self.common_network.train()
            self.specific_network_first.eval()
            self.specific_network_second.eval()

            for batch, ((S, targets1), (S2, targets2)) in enumerate(zipped_dataloader):
                
                S, targets1, S2, targets2 = S.squeeze().to(self.device), targets1.to(self.device), S2.squeeze().to(self.device), targets2.to(self.device) 
                
                self.common_network.zero_grad()

                loss_CU_task1 = self._loss_CU_task1(S, targets1=targets1) 
                mean_loss_cu1 += loss_CU_task1.item() / N

                loss_CU_task2 = self._loss_CU_task2(S2, targets2=targets2)
                mean_loss_cu2 += loss_CU_task2.item() / N

                loss = loss_CU_task1 + loss_CU_task2
                mean_loss_cu = mean_loss_cu1 + mean_loss_cu2

                loss.backward(retain_graph=True)
                optimizer_cu.step()

            # Training SU1    
            self.specific_network_first.train()
            self.common_network.eval()
            for batch, (S, targets1) in enumerate(first_loader):
                S, targets1 = S.squeeze().to(self.device), targets1.to(self.device)
                self.specific_network_first.zero_grad()
                loss_SU1 = self._loss_SU_first(S, targets1=targets1)
                mean_loss_su1 += loss_SU1.item() / N
                loss_SU1.backward()
                optimizer_su_first.step()


            # Training SU2
            self.specific_network_first.eval()
            self.common_network.eval()    
            self.specific_network_second.train()
            for batch, (S, targets2) in enumerate(second_loader):
                S, targets2 = S.squeeze().to(self.device), targets2.to(self.device)
                self.specific_network_second.zero_grad()
                loss_SU2 = self._loss_SU_second(S, targets2=targets2)
                mean_loss_su2 += loss_SU2.item() / N
                loss_SU2.backward()
                optimizer_su_second.step()


            # Training SU_ohnecu    
            
            for batch, (S, targets1) in enumerate(first_loader):
                S, targets1 = S.squeeze().to(self.device), targets1.to(self.device)
                self.specific_network_ohnecu.zero_grad()
                loss_SU_ohnecu = self._loss_SU_ohnecu(S, targets1=targets1)
                mean_loss_su_ohnecu += loss_SU_ohnecu.item() / N
                loss_SU_ohnecu.backward()
                optimizer_su_ohnecu.step()



            test_accuracy_task_one, test_accuracy_task_two, test_accuracy_task_one_ohnecu, error_rate_task1, error_rate_task2, error_rate_task1_ohnecu = self.eval(first_test_dataset, second_test_dataset)


            end = time.time()
            self.train_loss_su1.append(mean_loss_su1)
            self.train_loss_su2.append(mean_loss_su2)
            self.train_loss_su_ohnecu.append(mean_loss_su_ohnecu)
            self.train_loss_cu.append(mean_loss_cu)
            self.train_time.append(end - start)
            self.accuracy_task_1.append(test_accuracy_task_one)
            self.accuracy_task_2.append(test_accuracy_task_two)
            self.accuracy_task_ohnecu.append(test_accuracy_task_one_ohnecu)
            self.error_task_1.append(error_rate_task1)
            self.error_task_2.append(error_rate_task2)
            self.error_task_ohnecu.append(error_rate_task1_ohnecu)


            print(
                f"VAE epoch: {epoch_primal} / Train_Loss_CU: {mean_loss_cu:0.3f} / Train_Loss_SU1: {mean_loss_su1:0.3f} / Train_Loss_SU2: {mean_loss_su2:0.3f} / Train_Loss_SU_ohnecu: {mean_loss_su_ohnecu:0.3f}")
            
            print(
                f"Test Accuracy Task One: {test_accuracy_task_one:.2f}% / Test Accuracy Task Two: {test_accuracy_task_two:.2f}% / Test Accuracy Task ohnecu: {test_accuracy_task_one_ohnecu:.2f}%")





    def eval(self, first_test_dataset, second_test_dataset, batch_size= 100, threshold=0.5):

        first_test_loader = DataLoader(first_test_dataset, batch_size=batch_size, shuffle=False)
        second_test_loader = DataLoader(second_test_dataset, batch_size=batch_size, shuffle=False)

        self.common_network.eval()
        self.specific_network_first.eval()
        self.specific_network_second.eval()
        self.specific_network_ohnecu.eval()
        
        total_tone = 0
        correct_tone = 0

        total_ttwo = 0
        correct_ttwo = 0

        correct_tone_ohne = 0


        zipped_testloader = zip(first_test_loader, second_test_loader)

        with torch.no_grad():
            for batch ,((first_test_data, first_test_target),(second_test_data, second_test_target)) in enumerate(zipped_testloader):

                first_test_data = first_test_data.squeeze().to(self.device)
                first_test_target = first_test_target.to(self.device)
                second_test_target = second_test_target.to(self.device)

                c_mu, c_ln_var = self.common_network.cu_encode(first_test_data)
                c = reparameterize(c_mu, c_ln_var)
                # First Task
                x1_mu, x1_ln_var = self.specific_network_first.su_encode_first(c)
                x1 = reparameterize(x1_mu, x1_ln_var)
                x1_hat = awgn_channel(x1)
                z1     = self.specific_network_first.su_decode_first(x1_hat)
                z1_inferred = torch.sigmoid(z1)
                # Second Task
                x2_mu, x2_ln_var = self.specific_network_second.su_encode_second(c)
                x2 = reparameterize(x2_mu, x2_ln_var)
                x2_hat = awgn_channel(x2)
                z2 = self.specific_network_second.su_decode_second(x2_hat)
                z2_inferred = torch.softmax(z2, dim=1)
                # First Task Ohne CU
                x3_mu, x3_ln_var = self.specific_network_ohnecu.su_encode_ohnecu(first_test_data)
                x3 = reparameterize(x3_mu, x3_ln_var)
                x3_hat = awgn_channel(x3)
                z3     = self.specific_network_ohnecu.su_decode_ohnecu(x3_hat)
                z3_inferred = torch.sigmoid(z3)


                # Tests
                predicted_t1 = (z1_inferred > threshold).float()
                total_tone += first_test_target.size(0)
                first_test_target = first_test_target.view_as(predicted_t1)  # Ensuring target has the same shape as predicted
                correct_tone += torch.sum((predicted_t1 == first_test_target).float())   

                predicted_t2 = torch.argmax(z2_inferred, dim=1)
                total_ttwo += second_test_target.size(0)
                second_test_target = second_test_target.view_as(predicted_t2)  # Ensuring target has the same shape as predicted
                correct_ttwo += torch.sum((predicted_t2 == second_test_target).float())

                predicted_t3 = (z3_inferred > threshold).float()
                correct_tone_ohne += torch.sum((predicted_t3 == first_test_target).float())

        
        
        #print('Sum of First predictions:', predicted_t1.sum().item())
        accuracy_task_one = (correct_tone / total_tone) * 100
        error_task_one = 1 - (correct_tone / total_tone)
        #print('Accuracy of the First Task: {:.2f}%'.format(accuracy_task_one))

        #print('Sum of Second predictions:', predicted_t2.sum().item())
        accuracy_task_two = (correct_ttwo / total_ttwo) * 100
        error_task_two = 1 - (correct_ttwo / total_ttwo)
        #print('Accuracy of the Second Task: {:.2f}%'.format(accuracy_task_two))

        accuracy_task_one_ohnecu = (correct_tone_ohne / total_tone) * 100
        error_task_one_ohnecu = 1 - (correct_tone_ohne / total_tone)


        return accuracy_task_one, accuracy_task_two, accuracy_task_one_ohnecu, error_task_one, error_task_two, error_task_one_ohnecu