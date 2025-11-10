# stress testing
class ModelStressTester:

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

        # conversion en array le cas échéant
        self._convert_to_arrays()

        # gestion modèle
        if hasattr(model, '__class__'):
            self.model_name = model.__class__.__name__
        else:
            self.model_name = str(model)

        self.results = {}

    def _convert_to_arrays(self):
        """Conversion des données en array si dataframe"""
        if hasattr(self.X_test, 'values'):
            self.X_test = self.X_test.values
        if hasattr(self.y_test, 'values'):
            self.y_test = self.y_test.values

    def _ensure_array(self, data):
        """Vérification des array"""
        if hasattr(data, 'values'):
            return data.values
        return data

    def _get_predictions(self, X_data=None):
        if X_data is None:
            X_data = self.X_test
        else:
            X_data = self._ensure_array(X_data)

        try:
            if hasattr(self.model, "predict_proba"):
                try:
                    predictions = self.model.predict_proba(X_data)

                    if predictions.shape[1] == 2:
                        return predictions[:, 1]
                    else:
                        return predictions
                except:
                    return self.model.predict(X_data)

            elif hasattr(self.model, 'params'):
                try:
                    if X_data.shape[1] == len(self.model.params) - 1:
                        X_with_const = sm.add_constant(X_data, has_constant='add')
                    elif X_data.shape[1] == len(self.model.params):
                        X_with_const = X_data
                    else:
                        X_with_const = sm.add_constant(X_data, has_constant='add')

                    predictions = self.model.predict(X_with_const)
                    return predictions
                except Exception as e:
                    print(f"Erreur statsmodels: {e}")
                    return np.random.random(len(X_data))

            elif hasattr(self.model, "predict"):
                return self.model.predict(X_data)

            else:
                return np.random.random(len(X_data))

        except Exception as e:
            print(f"Erreur dans _get_predictions: {e}")
            return np.random.random(len(X_data))

    def _get_auc_score(self, predictions):
        """Calcul AUC"""
        try:
            if len(predictions) == 0:
                return 0.5

            if hasattr(predictions, 'shape') and len(predictions.shape) > 1 and predictions.shape[1] > 2:
                auc_scores = []
                for i in range(predictions.shape[1]):
                    try:
                        auc = roc_auc_score((self.y_test == i).astype(int), predictions[:, i])
                        auc_scores.append(auc)
                    except:
                        continue
                return np.mean(auc_scores) if auc_scores else 0.5

            elif np.all((predictions >= 0) & (predictions <= 1)):
                return roc_auc_score(self.y_test, predictions)

            else:
                binary_predictions = (predictions > 0.5).astype(int) if predictions.dtype.kind == 'f' else predictions
                return accuracy_score(self.y_test, binary_predictions)

        except Exception as e:
            print(f"Erreur dans _get_auc_score: {e}")
            return 0.5

    def noise_stress_test(self, noise_levels=[0.01, 0.05, 0.1, 0.2, 0.5], n_iter=10):
        """Test de robustesse au bruit"""
        print("=" * 60)
        print("STRESS TEST: BRUIT DANS LES FEATURES")
        print("=" * 60)

        baseline_predictions = self._get_predictions()
        baseline_auc = self._get_auc_score(baseline_predictions)
        print(f"AUC baseline (sans bruit): {baseline_auc:.4f}")

        noise_results = {}

        for noise_level in noise_levels:
            auc_scores = []

            for i in range(n_iter):
                try:
                    # Conversion en array pour les opérations numpy
                    X_test_array = self._ensure_array(self.X_test)

                    # Ajout de bruit gaussien
                    noise_std = np.std(X_test_array, axis=0)
                    noise_std = np.where(noise_std == 0, 1.0, noise_std)
                    noise = np.random.normal(0, noise_level * noise_std, X_test_array.shape)
                    X_noisy = X_test_array + noise

                    predictions_noisy = self._get_predictions(X_noisy)
                    auc = self._get_auc_score(predictions_noisy)
                    auc_scores.append(auc)
                except Exception as e:
                    print(f"Erreur itération {i} bruit {noise_level}: {e}")
                    auc_scores.append(0.5)

            if auc_scores:
                noise_results[noise_level] = {
                    'mean_auc': np.mean(auc_scores),
                    'std_auc': np.std(auc_scores),
                    'degradation': baseline_auc - np.mean(auc_scores),
                    'degradation_pct': ((baseline_auc - np.mean(auc_scores)) / baseline_auc) * 100 if baseline_auc > 0 else 0
                }

                print(f"Bruit {noise_level*100:.1f}% - AUC: {np.mean(auc_scores):.4f} "
                      f"(Dégradation: {noise_results[noise_level]['degradation_pct']:.1f}%)")

        self.results['noise_test'] = noise_results
        return noise_results

    def missing_data_stress_test(self, missing_ratios=[0.1, 0.2, 0.3, 0.5, 0.7], strategy='mean'):
        """Test de robustesse aux données manquantes"""
        print("\n" + "=" * 60)
        print("STRESS TEST: DONNÉES MANQUANTES")
        print("=" * 60)

        baseline_predictions = self._get_predictions()
        baseline_auc = self._get_auc_score(baseline_predictions)
        print(f"AUC baseline (données complètes): {baseline_auc:.4f}")

        missing_results = {}

        for ratio in missing_ratios:
            try:
                # conversion en array
                X_test_array = self._ensure_array(self.X_test).copy()

                # création données manquantes
                n_missing = int(X_test_array.size * ratio)

                # sélection aléatoire d'indices pour les valeurs manquantes
                missing_indices = np.random.choice(X_test_array.size, n_missing, replace=False)
                rows, cols = np.unravel_index(missing_indices, X_test_array.shape)

                # application valeurs manquantes
                X_test_array[rows, cols] = np.nan

                # imputation selon la stratégie
                if strategy == 'mean':
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='mean')
                    X_imputed = imputer.fit_transform(X_test_array)
                elif strategy == 'median':
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    X_imputed = imputer.fit_transform(X_test_array)
                else:
                    X_imputed = np.nan_to_num(X_test_array, nan=0)

                # prédictions sur données imputées
                predictions_imputed = self._get_predictions(X_imputed)
                auc = self._get_auc_score(predictions_imputed)

                missing_results[ratio] = {
                    'auc': auc,
                    'degradation': baseline_auc - auc,
                    'degradation_pct': ((baseline_auc - auc) / baseline_auc) * 100
                }

                print(f"Données manquantes {ratio*100:.0f}% - AUC: {auc:.4f} "
                      f"(Dégradation: {missing_results[ratio]['degradation_pct']:.1f}%)")
            except Exception as e:
                print(f"Erreur ratio {ratio}: {e}")
                missing_results[ratio] = {'auc': 0.5, 'degradation': 0, 'degradation_pct': 0}

        self.results['missing_data_test'] = missing_results
        return missing_results

    def data_drift_stress_test(self, drift_magnitudes=[0.1, 0.2, 0.5, 1.0, 2.0], n_iter=10):
        """Test de robustesse au data drift"""
        print("\n" + "=" * 60)
        print("STRESS TEST: DATA DRIFT")
        print("=" * 60)

        baseline_predictions = self._get_predictions()
        baseline_auc = self._get_auc_score(baseline_predictions)
        print(f"AUC baseline (sans drift): {baseline_auc:.4f}")

        drift_results = {}

        for magnitude in drift_magnitudes:
            auc_scores = []

            for i in range(n_iter):
                try:
                    # conversion en array
                    X_test_array = self._ensure_array(self.X_test)

                    # simulation de data drift
                    mean_shift = np.random.normal(0, magnitude, X_test_array.shape[1])
                    X_drifted = X_test_array + mean_shift

                    predictions_drifted = self._get_predictions(X_drifted)
                    auc = self._get_auc_score(predictions_drifted)
                    auc_scores.append(auc)
                except Exception as e:
                    print(f"Erreur itération {i} drift {magnitude}: {e}")
                    auc_scores.append(0.5)

            drift_results[magnitude] = {
                'mean_auc': np.mean(auc_scores),
                'std_auc': np.std(auc_scores),
                'degradation': baseline_auc - np.mean(auc_scores),
                'degradation_pct': ((baseline_auc - np.mean(auc_scores)) / baseline_auc) * 100
            }

            print(f"Drift magnitude {magnitude} - AUC: {np.mean(auc_scores):.4f} "
                  f"(Dégradation: {drift_results[magnitude]['degradation_pct']:.1f}%)")

        self.results['drift_test'] = drift_results
        return drift_results

    def adversarial_stress_test(self, attack_strengths=[0.01, 0.05, 0.1, 0.2], method='fgsm'):
        """Test de robustesse aux attaques adversaires"""
        print("\n" + "=" * 60)
        print("STRESS TEST: ATTAQUES ADVERSES")
        print("=" * 60)

        baseline_predictions = self._get_predictions()
        baseline_auc = self._get_auc_score(baseline_predictions)
        print(f"AUC baseline (sans attaque): {baseline_auc:.4f}")

        adversarial_results = {}

        for strength in attack_strengths:
            try:
                # Conversion en array
                X_test_array = self._ensure_array(self.X_test).copy()

                # Attaque simple
                if method == 'fgsm':
                    perturbation = strength * np.sign(np.random.randn(*X_test_array.shape))
                    X_attacked = X_test_array + perturbation

                predictions_attacked = self._get_predictions(X_attacked)
                auc = self._get_auc_score(predictions_attacked)

                adversarial_results[strength] = {
                    'auc': auc,
                    'degradation': baseline_auc - auc,
                    'degradation_pct': ((baseline_auc - auc) / baseline_auc) * 100
                }

                print(f"Attaque strength {strength} - AUC: {auc:.4f} "
                      f"(Dégradation: {adversarial_results[strength]['degradation_pct']:.1f}%)")
            except Exception as e:
                print(f"Erreur attaque {strength}: {e}")
                adversarial_results[strength] = {'auc': 0.5, 'degradation': 0, 'degradation_pct': 0}

        self.results['adversarial_test'] = adversarial_results
        return adversarial_results

    def outlier_stress_test(self, outlier_ratios=[0.01, 0.05, 0.1, 0.2], outlier_multiplier=10):
        """Test de robustesse aux outliers"""
        print("\n" + "=" * 60)
        print("STRESS TEST: OUTLIERS")
        print("=" * 60)

        baseline_predictions = self._get_predictions()
        baseline_auc = self._get_auc_score(baseline_predictions)
        print(f"AUC baseline (sans outliers): {baseline_auc:.4f}")

        outlier_results = {}

        for ratio in outlier_ratios:
            try:
                # conversion en array
                X_test_array = self._ensure_array(self.X_test).copy()
                n_outliers = int(len(X_test_array) * ratio)

                # sélection aléatoire d'échantillons
                outlier_indices = np.random.choice(len(X_test_array), n_outliers, replace=False)

                # création d'outliers
                X_test_array[outlier_indices] *= outlier_multiplier

                predictions_outliers = self._get_predictions(X_test_array)
                auc = self._get_auc_score(predictions_outliers)

                outlier_results[ratio] = {
                    'auc': auc,
                    'degradation': baseline_auc - auc,
                    'degradation_pct': ((baseline_auc - auc) / baseline_auc) * 100
                }

                print(f"Outliers {ratio*100:.1f}% - AUC: {auc:.4f} "
                      f"(Dégradation: {outlier_results[ratio]['degradation_pct']:.1f}%)")
            except Exception as e:
                print(f"Erreur outliers {ratio}: {e}")
                outlier_results[ratio] = {'auc': 0.5, 'degradation': 0, 'degradation_pct': 0}

        self.results['outlier_test'] = outlier_results
        return outlier_results

    def run_comprehensive_stress_test(self, tests_to_run=None):
        """
        Exécution des stress tests
        """
        print("=" * 80)
        print(f"STRESS TEST COMPLET - {self.model_name}")
        print("=" * 80)

        # tests par défaut
        if tests_to_run is None:
            tests_to_run = ['noise', 'missing', 'drift', 'adversarial', 'outlier']

        test_functions = {
            'noise': self.noise_stress_test,
            'missing': self.missing_data_stress_test,
            'drift': self.data_drift_stress_test,
            'adversarial': self.adversarial_stress_test,
            'outlier': self.outlier_stress_test
        }

        for test_name in tests_to_run:
            if test_name in test_functions:
                try:
                    test_functions[test_name]()
                except Exception as e:
                    print(f"Échec du test {test_name}: {e}")

        # rapport général
        self.generate_stress_report()

    def generate_stress_report(self):
        """
        Rapport visuel complet des stress tests
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Rapport de Stress Testing - {self.model_name}', fontsize=16, fontweight='bold')

        # bruit
        if 'noise_test' in self.results:
            noise_data = self.results['noise_test']
            x = list(noise_data.keys())
            y = [noise_data[level]['mean_auc'] for level in x]
            axes[0, 0].plot(x, y, 'o-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Niveau de bruit')
            axes[0, 0].set_ylabel('AUC')
            axes[0, 0].set_title('Robustesse au bruit')
            axes[0, 0].grid(True, alpha=0.3)

        # valeurs manquantes
        if 'missing_data_test' in self.results:
            missing_data = self.results['missing_data_test']
            x = list(missing_data.keys())
            y = [missing_data[ratio]['auc'] for ratio in x]
            axes[0, 1].plot(x, y, 'o-', linewidth=2, markersize=8, color='orange')
            axes[0, 1].set_xlabel('Ratio données manquantes')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].set_title('Robustesse aux données manquantes')
            axes[0, 1].grid(True, alpha=0.3)

        # data drift
        if 'drift_test' in self.results:
            drift_data = self.results['drift_test']
            x = list(drift_data.keys())
            y = [drift_data[magnitude]['mean_auc'] for magnitude in x]
            axes[0, 2].plot(x, y, 'o-', linewidth=2, markersize=8, color='green')
            axes[0, 2].set_xlabel('Magnitude du drift')
            axes[0, 2].set_ylabel('AUC')
            axes[0, 2].set_title('Robustesse au data drift')
            axes[0, 2].grid(True, alpha=0.3)

        # attaques adverses
        if 'adversarial_test' in self.results:
            adv_data = self.results['adversarial_test']
            x = list(adv_data.keys())
            y = [adv_data[strength]['auc'] for strength in x]
            axes[1, 0].plot(x, y, 'o-', linewidth=2, markersize=8, color='red')
            axes[1, 0].set_xlabel('Force attaque')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].set_title('Robustesse aux attaques adverses')
            axes[1, 0].grid(True, alpha=0.3)

        # outliers
        if 'outlier_test' in self.results:
            outlier_data = self.results['outlier_test']
            x = list(outlier_data.keys())
            y = [outlier_data[ratio]['auc'] for ratio in x]
            axes[1, 1].plot(x, y, 'o-', linewidth=2, markersize=8, color='purple')
            axes[1, 1].set_xlabel('Ratio outliers')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].set_title('Robustesse aux outliers')
            axes[1, 1].grid(True, alpha=0.3)

        # résumé
        axes[1, 2].axis('off')
        summary_text = "RÉSUMÉ DES STRESS TESTS\n\n"

        baseline_predictions = self._get_predictions()
        baseline_auc = self._get_auc_score(baseline_predictions)
        summary_text += f"AUC Baseline: {baseline_auc:.4f}\n\n"

        for test_name, test_data in self.results.items():
            worst_case = min(test_data.values(), key=lambda x: x.get('auc', x.get('mean_auc', 1)))
            worst_auc = worst_case.get('auc', worst_case.get('mean_auc', 0))
            degradation_pct = ((baseline_auc - worst_auc) / baseline_auc) * 100
            test_name_display = test_name.replace('_test', '').replace('_', ' ').title()
            summary_text += f"{test_name_display}:\n  Pire AUC: {worst_auc:.4f}\n  Dégradation: {degradation_pct:.1f}%\n\n"

        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10)

        plt.tight_layout()
        plt.show()

        return fig
