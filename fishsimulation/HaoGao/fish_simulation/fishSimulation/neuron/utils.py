import torch

def synaptic_model4(self, possion):
    J_ui_activate_part = torch.ones(4, 1) * possion
    self.J_ui = self.J_ui * torch.exp(-self.delta_t / self.tau_ui)
    self.J_ui += J_ui_activate_part

    I_ui = self.g_ui * (self.V_ui - self.V_i) * self.J_ui
    I_syn = I_ui.sum(dim=0)

    return I_syn