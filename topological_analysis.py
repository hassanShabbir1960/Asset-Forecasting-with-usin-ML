class TopologicalAnalysis:
    def __init__(self, dataset_TI_df, target_col):
        self.dataset_TI_df = dataset_TI_df
        self.target_col = target_col

    def preprocess_data(self):
        # Preprocessing steps
        self.dataset_TI_df[['Date', self.target_col]].set_index('Date')
        start_year = '2008'
        self.price_resampled_df = self.dataset_TI_df[['Date', self.target_col]].set_index('Date').resample('24H').pad()[start_year:]

    def apply_topological_analysis(self):
        # Topological analysis steps
        embedding_dimension = 3
        embedding_time_delay = 2
        embedder = ts.TakensEmbedding(
            parameters_type="fixed",
            dimension=embedding_dimension,
            time_delay=embedding_time_delay,
            n_jobs=-1,
        )
        price_values = self.price_resampled_df.values
        price_embedded = embedder.fit_transform(price_values)
        embedder_time_delay = embedder.time_delay_
        embedder_dimension = embedder.dimension_
        window_width = 30
        window_stride = 4
        sliding_window = ts.SlidingWindow(width=window_width, stride=window_stride)
        price_embedded_windows = sliding_window.fit_transform(price_embedded)

        homology_dimensions = (0, 1)
        VR = hl.VietorisRipsPersistence(homology_dimensions=homology_dimensions, n_jobs=1)
        self.diagrams = VR.fit_transform(price_embedded_windows)

    def post_process_data(self):
        # Post-processing steps
        indices = [win[1] - 1 for win in window_indices[1:]]
        time_index_derivs = self.price_resampled_df.iloc[indices].index
        resampled_close_price_derivs = self.price_resampled_df.loc[time_index_derivs]
        resampled_close_price_derivs['betti_succ_dists'] = betti_succ_dists
        resampled_close_price_derivs['landscape_succ_dists'] = landscape_succ_dists
        self.dataset_TI_df = self.dataset_TI_df.merge(resampled_close_price_derivs, left_on='Date', right_on='Date', how='left')
        self.dataset_TI_df[['betti_succ_dists', 'landscape_succ_dists']] = self.dataset_TI_df[['betti_succ_dists', 'landscape_succ_dists']].fillna(method='ffill')

        return self.dataset_TI_df

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def add_dfi(self, dfi_file_path):
        DFI = pd.read_csv(dfi_file_path)
        DFI.rename(columns={'Close': 'close'}, inplace=True)
        DFI = self.get_technical_indicators(DFI)
        DFI.columns = [str(col) + '_dfi' for col in DFI.columns]
        DFI['Date_dfi'] = pd.to_datetime(DFI['Date_dfi'])
        self.dataset_TI_df = self.dataset_TI_df.merge(DFI, left_on='Date', right_on='Date_dfi', how='left').drop('Date_dfi', axis=1)
        self.dataset_TI_df.fillna(0.0, inplace=True)
    
