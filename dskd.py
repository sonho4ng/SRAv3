
# def compute_forward_kl_divergence(logits, teacher_logits, kd_temp=2.0):
#     logits = logits / kd_temp
#     teacher_logits = teacher_logits / kd_temp
    
#     lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
#     teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
#     teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
#     kld = (teacher_probs * (teacher_lprobs - lprobs))
#     inf_mask = logits.isinf()
#     kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

#     return kld

# def compute_skewed_reverse_kl_divergence(logits, teacher_logits, kd_temp=2.0, skew_lambda=0.1):
#         logits = logits / kd_temp
#         teacher_logits = teacher_logits / kd_temp

#         student_probs = torch.softmax(logits, -1, dtype=torch.float32)
#         teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
#         mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * student_probs
#         mixed_lprobs = torch.log(mixed_probs)
#         student_lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
#         kld = (student_probs * (student_lprobs - mixed_lprobs))
#         inf_mask = logits.isinf() | teacher_logits.isinf()
#         kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

#         return kld    

# def compute_dual_space_kd_loss_with_cma(
#         self, outputs, teacher_outputs, output_data, student_model, teacher_model, projectors
#     ):
#         target = output_data["label"]
#         teacher_target = output_data[f"teacher_label"]
        
        
#         pad_mask = target.ne(-100)
#         teacher_pad_mask = teacher_target.ne(-100)

#         hiddens = outputs.hidden_states[-1]
#         teacher_hiddens = teacher_outputs.hidden_states[-1]
        
#         t2s_hiddens = projectors["t2s"](teacher_hiddens)
#         t2s_logits = t2s_hiddens.matmul(student_model.lm_head.weight.detach().transpose(-1, -2))

#         t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)
#         t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        
#         t2s_kd_loss = compute_skewed_reverse_kl_divergence(outputs.logits, t2s_logits.detach())
#         t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum()

#         s2t_hiddens = hiddens
#         s2t_logits = s2t_hiddens.matmul(teacher_model.lm_head.weight.detach().transpose(-1, -2))

#         s2t_kd_loss = compute_forward_kl_divergence(s2t_logits, teacher_outputs.logits)
#         s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum()
        
#         kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss

#         # else:
#         #     kd_loss = t2s_ce_loss

#         return kd_loss
    

