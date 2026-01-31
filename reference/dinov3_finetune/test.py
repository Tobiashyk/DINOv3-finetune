# def ibot_loss(student_patches, teacher_patches, mask_indices, temperature=0.1):
#     batch_size = student_patches.size(0)
#     h_patches = w_patches = int(student_patches.size(1) ** 0.5)
#     total_loss = 0.0
#     total_batches = 0

#     for b in range(batch_size):
#         indices_b = mask_indices[b]
#         if indices_b.numel() == 0:
#             continue
        
#         # get masked patch tokens
#         linear_indices = indices_b[:, 0] * w_patches + indices_b[:, 1]
#         student_all = F.normalize(student_patches[b], dim=-1)  # [num_patches, emb_dim]
#         teacher_all = F.normalize(teacher_patches[b], dim=-1)  # [num_patches, emb_dim]

#         student_masked_patches = student_all[linear_indices]
#         teacher_masked_patches = teacher_all[linear_indices]

#         # compute logits
#         student_logits = student_masked_patches / temperature
#         teacher_logits = teacher_masked_patches / (temperature - 0.03)

#         # compute probabilities
#         student_log_probs = F.log_softmax(student_logits, dim=-1)
#         teacher_probs = F.softmax(teacher_logits, dim=-1)

#         # compute loss
#         loss = torch.sum(-teacher_probs * student_log_probs, dim=-1).mean()

#         total_loss += loss
#         total_batches += 1

#     ibot_loss = total_loss / total_batches
#     return ibot_loss

# def ibot_loss(student_patches, teacher_patches, mask_indices, temperature=0.1):
#     batch_size = student_patches.size(0)
#     h_patches = w_patches = int(student_patches.size(1) ** 0.5)
#     total_loss = 0.0
#     total_batches = 0

#     for b in range(batch_size):
#         indices_b = mask_indices[b]
#         if indices_b.numel() == 0:
#             continue
        
#         # get masked patch tokens
#         linear_indices = indices_b[:, 0] * w_patches + indices_b[:, 1]
#         student_all = F.normalize(student_patches[b], dim=-1)  # [num_patches, emb_dim]
#         teacher_all = F.normalize(teacher_patches[b], dim=-1)  # [num_patches, emb_dim]

#         student_masked_patches = student_all[linear_indices]

#         sim_matrix = torch.matmul(student_masked_patches, teacher_all.T) / temperature
#         labels = linear_indices

#         loss = F.cross_entropy(sim_matrix, labels)

#         total_loss += loss
#         total_batches += 1

#     ibot_loss = total_loss / total_batches
#     return ibot_loss