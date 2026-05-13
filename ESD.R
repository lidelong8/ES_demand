# 加载必要的包
library(readxl)
library(xgboost)
library(iml)
library(ggplot2)
library(caret)
library(patchwork)

file_path <- "E:/0510gaitu/Data_process_rf_20240929.xls"
output_dir <- "E:/0510gaitu/"

sheets <- c('drivers', 'AP-dominated', 'RE-dominated', 'Balanced')

# 变量名称
feature_names_clean <- c('Gender', 'Age', 'Edu.', 'Income', 'Interest', 'Frequency', 'Sizes',
                         'PM2.5', 'VegCover', 'GDP', 'POP', 'Precip', 'Temp.')

cls <- c(1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2)

# 离散变量列表
discrete_vars <- c('Gender', 'Age', 'Edu.', 'Income', 'Frequency', 'Sizes')

subplot_labels <- paste0("(", letters[1:14], ")")
BASE_SIZE <- 28

for (sheet_name in sheets) {
  print(paste("Processing sheet:", sheet_name))
  
  df <- read_excel(file_path, sheet = sheet_name)
  y  <- df[[1]]
  X  <- df[, 2:14]
  colnames(X) <- feature_names_clean
  
  for (dv in discrete_vars) {
    X[[dv]] <- as.integer(round(X[[dv]]))
  }
  
  set.seed(0)
  
  train_index <- createDataPartition(y, p = 0.9, list = FALSE)
  X_train     <- as.data.frame(X[train_index, ])
  y_train     <- y[train_index]
  X_train_mat <- data.matrix(X_train)
  
  X_train_mat_unnamed <- X_train_mat
  colnames(X_train_mat_unnamed) <- NULL
  dtrain <- xgb.DMatrix(data = X_train_mat_unnamed, label = y_train)
  
  params <- list(
    objective        = "reg:squarederror",
    eta              = 0.1,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8
  )
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 200, verbose = 0)
  
  # 计算模型的平均预测值（基准值），用于平移 ALE 使其全为正数
  base_value <- mean(predict(xgb_model, X_train_mat_unnamed))
  print(paste("  Base average prediction:", round(base_value, 4)))
  
  predict_fn <- function(model, newdata) {
    mat <- data.matrix(newdata)
    colnames(mat) <- NULL
    predict(model, mat)
  }
  
  predictor <- Predictor$new(
    model      = xgb_model,
    data       = X_train,
    y          = y_train,
    predict.function = predict_fn
  )
  
  ale_list <- list()
  ale_importance <- numeric(13)
  names(ale_importance) <- feature_names_clean
  
  for (j in 1:13) {
    feat <- feature_names_clean[j]
    
    ale_obj <- FeatureEffect$new(
      predictor,
      feature = feat,
      method  = "ale"
    )
    
    ale_data <- ale_obj$results
    ale_list[[feat]] <- ale_data
    
    ale_importance[feat] <- mean(abs(ale_data$.value), na.rm = TRUE)
  }
  
  # 严格按照重要性排序
  sorted_idx <- order(ale_importance, decreasing = TRUE)
  sorted_names <- names(ale_importance)[sorted_idx]
  
  imp_df <- data.frame(
    Feature    = factor(sorted_names, levels = sorted_names),
    Importance = ale_importance[sorted_idx],
    Type       = ifelse(cls[sorted_idx] == 1,
                        "Socio-economic or personal variables",
                        "Environmental variables")
  )
  
  titl <- ifelse(grepl("drivers", sheet_name), "All", sheet_name)
  
  # 绘制重要性柱状图 (a)
  p_imp <- ggplot(imp_df, aes(x = Feature, y = Importance, fill = Type)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("Socio-economic or personal variables" = "blue",
                                 "Environmental variables" = "red")) +
    labs(title = titl, y = "Variable Importance", x = "",
         tag = subplot_labels[1]) +
    theme_classic(base_size = BASE_SIZE) +
    theme(axis.text.x       = element_text(angle = 20, hjust = 1, face = "bold"),
          axis.title        = element_text(face = "bold"),
          legend.position   = c(0.75, 0.9),
          legend.title      = element_blank(),
          legend.text       = element_text(size = BASE_SIZE * 0.8),
          plot.title        = element_text(face = "bold", size = BASE_SIZE * 1.2),
          plot.tag          = element_text(size = BASE_SIZE, face = "bold"),
          plot.tag.position = "topleft")
  
  dep_plots <- list()
  
  # 统一循环，严格按照 sorted_names (重要性) 顺序绘制子图
  for (i in 1:13) {
    feat_name  <- sorted_names[i]
    feat_idx   <- which(feature_names_clean == feat_name)
    feat_color <- ifelse(cls[feat_idx] == 1, "#3498db", "#e74c3c")
    
    ale_data  <- ale_list[[feat_name]]
    cur_label <- subplot_labels[i + 1] # 标签从 (b) 开始
    
    # 控制 Y 轴标签显示：第1个图(右上角) 和 每行第1个图(左侧) 显示 Y 轴标签
    show_y <- (i == 1) | (i %% 3 == 2)
    
    if (feat_name == "Gender") {
      plot_df <- data.frame(
        x    = ale_data[[feat_name]],
        yval = ale_data$.value + base_value
      )
      plot_df$x_pos <- ifelse(plot_df$x == 2, 3, plot_df$x)
      
      p_dep <- ggplot(plot_df, aes(x = x_pos, y = yval)) +
        geom_col(fill = feat_color, width = 0.6, alpha = 0.8) +
        scale_x_continuous(limits = c(0, 4), breaks = c(1, 3), labels = c("male", "female")) +
        theme_classic(base_size = BASE_SIZE) +
        labs(x = feat_name, y = ifelse(show_y, "Trade-off Intensity", ""), tag = cur_label) +
        theme(axis.title = element_text(face = "bold"),
              plot.tag = element_text(size = BASE_SIZE, face = "bold"), plot.tag.position = "topleft")
      
      dep_plots[[i]] <- p_dep
      
    } else if (feat_name %in% discrete_vars) {
      plot_df <- data.frame(
        x    = ale_data[[feat_name]],
        yval = ale_data$.value + base_value
      )
      x_breaks <- sort(unique(plot_df$x))
      
      p_dep <- ggplot(plot_df, aes(x = x, y = yval)) +
        geom_step(color = feat_color, linewidth = 1.2) +
        geom_point(color = feat_color, size = 3) +
        tryCatch(
          geom_smooth(method = "loess", formula = y ~ x, color = "black", linetype = "dashed", linewidth = 1.0, se = FALSE, span = 1.0),
          error = function(e) geom_smooth(method = "lm", formula = y ~ x, color = "black", linetype = "dashed", linewidth = 1.0, se = FALSE)
        ) +
        scale_x_continuous(breaks = x_breaks) +
        theme_classic(base_size = BASE_SIZE) +
        labs(x = feat_name, y = ifelse(show_y, "Trade-off Intensity", ""), tag = cur_label) +
        theme(axis.title = element_text(face = "bold"),
              plot.tag = element_text(size = BASE_SIZE, face = "bold"), plot.tag.position = "topleft")
      
      dep_plots[[i]] <- p_dep
      
    } else {
      plot_df <- data.frame(
        x    = ale_data[[feat_name]],
        yval = ale_data$.value + base_value
      )
      
      p_dep <- ggplot(plot_df, aes(x = x, y = yval)) +
        geom_line(color = feat_color, linewidth = 1.0) +
        geom_smooth(method = "loess", formula = y ~ x, color = "black", linetype = "dashed", linewidth = 1.0, se = FALSE, span = 0.75) +
        theme_classic(base_size = BASE_SIZE) +
        labs(x = feat_name, y = ifelse(show_y, "Trade-off Intensity", ""), tag = cur_label) +
        theme(axis.title = element_text(face = "bold"),
              plot.tag = element_text(size = BASE_SIZE, face = "bold"), plot.tag.position = "topleft")
      
      dep_plots[[i]] <- p_dep
    }
  }
  
  # 拼图设计：A 占两格，B~N 依次排列
  design <- "
    AAB
    CDE
    FGH
    IJK
    LMN
  "
  
  # 将主图和所有子图按顺序合并
  all_plots  <- c(list(p_imp), dep_plots)
  final_plot <- wrap_plots(all_plots, design = design)
  
  save_path <- paste0(output_dir, "XGBoost_ALE_Positive_Ordered_", titl, ".jpg")
  ggsave(save_path, plot = final_plot, width = 28, height = 24, dpi = 350)
  
  print(paste("Finished and saved:", save_path))
}
print("All Finished!")