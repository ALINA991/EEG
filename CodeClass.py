import Class_sf_perm

#score_clf_DA, perm_score_clf_DA, pval_clf_DA, score_knn_DA, perm_score_knn_DA, pval_knn_DA, score_lda_DA, perm_score_lda_DA, pval_lda_DA, score_qda_DA, perm_score_qda_DA, pval_qda_DA, score_mlp_DA, perm_score_mlp_DA, pval_mlp_DA= Class2.class_sf_perm('DA') 

#score_clf_LA, perm_score_clf_LA, pval_clf_LA, score_knn_LA, perm_score_knn_LA, pval_knn_LA, score_lda_LA, perm_score_lda_LA, pval_lda_LA, score_qda_LA, perm_score_qda_LA, pval_qda_LA, score_mlp_LA, perm_score_mlp_LA, pval_mlp_LA= Class2.class_sf_perm('LA') 

score_clf_DA, perm_score_clf_DA, pval_clf_DA= Class_sf_perm.class_clf('DA')
score_knn_DA, perm_score_knn_DA, pval_knn_DA= Class_sf_perm.class_knn('DA')
score_lda_DA, perm_score_lda_DA, pval_lda_DA= Class_sf_perm.class_lda('DA')
score_qda_DA, perm_score_qda_DA, pval_qda_DA= Class_sf_perm.class_qda('DA')
score_mlp_DA, perm_score_mlp_DA, pval_mlp_DA= Class_sf_perm.class_mlp('DA')
