# création d'histogramme empilés
def plot_stacked_hist(df, x_col, stack_col, percent=False, title=None):
    # vérifications
    if x_col not in df.columns or stack_col not in df.columns:
        raise ValueError("Les colonnes spécifiées n'existent pas")

    # tableau croisé des valeurs
    table = pd.crosstab(df[x_col], df[stack_col])

    # conversion en pourcentage le cas échéant
    if percent:
        table = table.div(table.sum(axis=1), axis=0) * 100
        ylabel = "Pourcentage (%)"
    else:
        ylabel = "Nombre"

    # graphique
    ax = table.plot(kind='bar', stacked=True, figsize=(8, 5))
    plt.title(title or f"{stack_col} par {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.legend(title=stack_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
  
# création de pie charts
def plot_pie(df, col, title=None, autopct='%1.1f%%', startangle=90):
    if col not in df.columns:
        raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

    counts = df[col].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(
        counts,
        labels=counts.index,
        autopct=autopct,
        startangle=startangle,
        wedgeprops={'edgecolor': 'white'}
    )
    plt.title(title or f"Répartition de {col}")
    plt.tight_layout()
    plt.show()

# calcul du WoE (weight of evidence) et IV (information value) avec visualisations graphiques
def calculate_woe_iv(df, feature, target, min_perc=0.05):
    """ Calcul WoE et IV """

    # vérifications
    if feature not in df.columns:
        raise ValueError(f"La variable {feature} n'existe pas dans le DataFrame")
    if target not in df.columns:
        raise ValueError(f"La variable cible {target} n'existe pas dans le DataFrame")

    # copie données
    temp_df = df[[feature, target]].copy()

    # création de bins pour les variables continues
    if temp_df[feature].dtype in ['float64', 'int64']:
        n_bins = min(10, len(temp_df[feature].unique())) # usage des quantiles
        temp_df['bin'] = pd.qcut(temp_df[feature], q=n_bins, duplicates='drop')
    else:
        temp_df['bin'] = temp_df[feature]

    grouped = temp_df.groupby('bin', observed=True).agg({
        target: ['count', 'sum']
    })
    grouped.columns = ['total', 'events']
    grouped = grouped.reset_index()

    # calcul
    grouped['non_events'] = grouped['total'] - grouped['events']

    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()

    # pourcentages
    grouped['pct_events'] = grouped['events'] / total_events
    grouped['pct_non_events'] = grouped['non_events'] / total_non_events

    # éviter division par zéro le cas échéant
    grouped['pct_events'] = grouped['pct_events'].replace(0, 0.0001)
    grouped['pct_non_events'] = grouped['pct_non_events'].replace(0, 0.0001)

    # calcul du WoE
    grouped['woe'] = np.log(grouped['pct_events'] / grouped['pct_non_events'])

    # calcul de l'IV pour chaque bin
    grouped['iv_component'] = (grouped['pct_events'] - grouped['pct_non_events']) * grouped['woe']

    # IV total
    iv_total = grouped['iv_component'].sum()

    # taux d'événement
    grouped['event_rate'] = grouped['events'] / grouped['total']

    # ordre des bins pour l'affichage de résultats
    if temp_df[feature].dtype in ['float64', 'int64']:
        grouped = grouped.sort_values('bin')

    # dataframe des résultats
    woe_iv_df = grouped[[
        'bin', 'total', 'events', 'non_events',
        'event_rate', 'woe', 'iv_component'
    ]].copy()

    return woe_iv_df, iv_total

def analyze_all_variables_woe_iv(df, target_variable, exclude_vars=None, min_perc=0.05):
    """ Analyse du WoE et de l'IV pour toutes les variables """

    if exclude_vars is None:
        exclude_vars = [target_variable]
    else:
        exclude_vars = exclude_vars + [target_variable]

    # variables à analyser
    variables = [col for col in df.columns if col not in exclude_vars]

    results = {}
    iv_summary = []

    print("=" * 80)
    print("📊 ANALYSE WOE ET IV")
    print("=" * 80)

    for var in variables:
        try:
            woe_df, iv = calculate_woe_iv(df, var, target_variable, min_perc)
            results[var] = woe_df

            # interprétation de l'IV
            if iv < 0.02:
                power = "Non prédictive"
            elif iv < 0.1:
                power = "Faible"
            elif iv < 0.3:
                power = "Moyenne"
            elif iv < 0.5:
                power = "Forte"
            else:
                power = "Suspecte"

            iv_summary.append({
                'Variable': var,
                'IV': iv,
                'Power': power,
                'Type': df[var].dtype
            })

            print(f"{var:25} | IV = {iv:7.4f} | {power:15} | {str(df[var].dtype):10}")

        except Exception as e:
            print(f"{var:25} | Erreur: {e}")
            iv_summary.append({
                'Variable': var,
                'IV': np.nan,
                'Power': 'Erreur',
                'Type': df[var].dtype
            })

    # dataframe récapitulatif
    iv_summary_df = pd.DataFrame(iv_summary).sort_values('IV', ascending=False)

    print(f"\n Récapitulatif IV - {len(variables)} variables analysées")
    print("=" * 80)

    return results, iv_summary_df

def plot_woe_analysis(results, variable, figsize=(12, 8)):
    """
    Graphique d'analyse WoE pour une variable
    """
    if variable not in results:
        print(f"Variable {variable} non trouvée dans les résultats")
        return

    woe_df = results[variable]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # graphique 1 : WoE par bin
    ax1.bar(range(len(woe_df)), woe_df['woe'], color='skyblue', edgecolor='navy')
    ax1.set_xticks(range(len(woe_df)))
    ax1.set_xticklabels([str(b) for b in woe_df['bin']], rotation=45)
    ax1.set_ylabel('Weight of Evidence (WoE)')
    ax1.set_title(f'WoE par bin - {variable}')
    ax1.grid(axis='y', alpha=0.3)

    # ajout des valeurs sur les barres
    for i, v in enumerate(woe_df['woe']):
        ax1.text(i, v + (0.1 if v >= 0 else -0.15), f'{v:.2f}',
                ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')

    # graphique 2 : Taux d'événements par bin
    ax2.bar(range(len(woe_df)), woe_df['event_rate'] * 100,
            color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax2.set_xticks(range(len(woe_df)))
    ax2.set_xticklabels([str(b) for b in woe_df['bin']], rotation=45)
    ax2.set_ylabel('Taux d\'événements (%)')
    ax2.set_xlabel('Bins')
    ax2.grid(axis='y', alpha=0.3)

    # ajout des pourcentages
    for i, v in enumerate(woe_df['event_rate'] * 100):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # affichage tableau détaillée
    print(f"\n Détail WoE et IV - {variable}")
    print("=" * 60)
    display_df = woe_df.copy()
    display_df["event_rate"] = (display_df['event_rate'] * 100).round(1).astype(str) + '%'
    display_df["woe"] = display_df['woe'].round(3)
    display_df['iv_component'] = display_df['iv_component'].round(4)
    print(display_df.to_string(index=False))

def plot_iv_summary(iv_summary_df, figsize=(10, 6)):
    """ Graphique récapitulatif des IV """

    plt.figure(figsize=figsize)

    # filtrage des variables avec un IV valide
    valid_iv = iv_summary_df[iv_summary_df['IV'].notna()].sort_values('IV')

    # couleurs selon la puissance discriminative
    colors = {
        'Non prédictive': 'lightgray',
        'Faible': 'yellow',
        'Moyenne': 'orange',
        'Forte': 'red',
        'Suspecte': 'darkred'
    }

    bar_colors = [colors.get(power, 'gray') for power in valid_iv['Power']]

    bars = plt.barh(valid_iv['Variable'], valid_iv['IV'], color=bar_colors, edgecolor='black')

    plt.xlabel('Information Value (IV)')
    plt.title('Power Prédictive des Variables (Information Value)')
    plt.grid(axis='x', alpha=0.3)

    # ajout des valeurs d'IV
    for i, (iv, power) in enumerate(zip(valid_iv['IV'], valid_iv['Power'])):
        plt.text(iv + 0.01, i, f'{iv:.3f}', va='center', fontweight='bold')

    # légende pour les seuils d'IV
    plt.axvline(x=0.02, color='red', linestyle='--', alpha=0.5, label='IV=0.02 (Faible)')
    plt.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='IV=0.1 (Moyen)')
    plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.5, label='IV=0.3 (Fort)')

    plt.legend()
    plt.tight_layout()
    plt.show()

# implémentation modèles
def Implementation_model(model, X_train, y_train, hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {}

    if hasattr(model, "predict_proba"):
        sklearn_model = model(**hyperparameters)
        result = sklearn_model.fit(X_train, y_train)
    else:
        X_train_sm = sm.add_constant(X_train)
        logit_model = model(y_train, X_train_sm)
        result = logit_model.fit(disp=False)

    return result

# graphiques courbes ROC (receiver operating characteristic) et coefficient AUC (area under the curve) 
def plot_roc_auc(y_true, y_pred_proba, title = "Base test", model_name="Modèle", figsize=(10, 6)):
    # vérification inputs
    if y_true is None or y_pred_proba is None:
        raise ValueError("y_true et y_pred_proba ne peuvent pas être None")

    if len(y_true) != len(y_pred_proba):
        raise ValueError("y_true et y_pred_proba doivent avoir la même longueur")

    # calcul métriques liés à la courbe ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # calcul coefficient AUC
    auc_score = roc_auc_score(y_true, y_pred_proba)

    # courbe ROC
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC {model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Aléatoire (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (False Positive Rate)')
    plt.ylabel('Taux de Vrais Positifs (True Positive Rate)')
    plt.title(f'Courbe ROC - {model_name} ({title})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.text(0.6, 0.3, f'AUC = {roc_auc:.3f}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()

    # affichage explicit des résultats
    print(f"Courbe ROC générée pour {model_name}")
    print(f"Score AUC: {auc_score:.4f}")

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
        'auc_sklearn': auc_score
    }
# def bootstrap_auc_histogram(model, X_test, y_test, bootstrap_fraction=0.05,
                           n_bootstrap=1000, random_state=42, figsize=(12, 8)):
    """
    Réalise un bootstrap sur un pourcentage de la base test et affiche l'histogramme des AUC
    """

    print("=" * 60)
    print("BOOTSTRAP AUC ANALYSIS")
    print("=" * 60)

    # calcul taille échantillon bootstrap
    sample_size = int(len(X_test) * bootstrap_fraction)

    print(f"Taille totale du test set: {len(X_test)}")
    print(f"Taille des échantillons bootstrap: {sample_size} ({bootstrap_fraction*100}%)")
    print(f"Nombre d'itérations bootstrap: {n_bootstrap}")

    # stockage des scores AUC
    auc_scores = []

    # boucle bootstrap
    for i in range(n_bootstrap):
        # échnantillonnage avec remise
        X_bootstrap, y_bootstrap = resample(
            X_test, y_test,
            n_samples=sample_size,
            replace=True,
            random_state=random_state + i,
            stratify=y_test
        )

        if not hasattr(model, 'predict_proba'):
        # prédictions
          y_pred_proba = model.predict(X_bootstrap)

        else:
          y_pred_proba = model.predict_proba(X_bootstrap)[:,1]

        auc = roc_auc_score(y_bootstrap,y_pred_proba)

        auc_scores.append(auc)

    # conversion en array
    auc_scores = np.array(auc_scores)

    # calcul des statistiques
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    median_auc = np.median(auc_scores)
    ci_lower = np.percentile(auc_scores, 2.5)
    ci_upper = np.percentile(auc_scores, 97.5)

    # affichage des statistiques
    print(f"\nSTATISTIQUES DES SCORES AUC:")
    print(f"Moyenne: {mean_auc:.4f}")
    print(f"Écart-type: {std_auc:.4f}")
    print(f"Médiane: {median_auc:.4f}")
    print(f"Intervalle de confiance 95%: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Minimum: {np.min(auc_scores):.4f}")
    print(f"Maximum: {np.max(auc_scores):.4f}")

    # création de l'histogramme
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # histogramme principal
    n, bins, patches = ax1.hist(auc_scores, bins=50, alpha=0.7, color='skyblue',
                               edgecolor='black', density=True)

    # lignes verticales pour les statistiques
    ax1.axvline(mean_auc, color='red', linestyle='--', linewidth=2,
                label=f'Moyenne: {mean_auc:.4f}')
    ax1.axvline(median_auc, color='green', linestyle='--', linewidth=2,
                label=f'Médiane: {median_auc:.4f}')
    ax1.axvline(ci_lower, color='orange', linestyle=':', linewidth=1.5,
                label=f'IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax1.axvline(ci_upper, color='orange', linestyle=':', linewidth=1.5)

    ax1.set_xlabel('Score AUC')
    ax1.set_ylabel('Densité')
    ax1.set_title(f'Distribution des scores AUC - Bootstrap {bootstrap_fraction*100}% du test set\n'
                 f'{n_bootstrap} itérations - {sample_size} échantillons par itération')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # box plot
    ax2.boxplot(auc_scores, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Score AUC')
    ax2.set_title('Boxplot des scores AUC')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return auc_scores, fig

# fonction pour analyse détaillée
def detailed_bootstrap_analysis(auc_scores, model_name="Modèle"):
    """
    Analyse détaillée des résultats bootstrap
    """
    print("\n" + "=" * 50)
    print(f"ANALYSE DÉTAILLÉE - {model_name.upper()}")
    print("=" * 50)

    # Statistiques descriptives
    from scipy import stats

    stats_dict = {
        'Moyenne': np.mean(auc_scores),
        'Médiane': np.median(auc_scores),
        'Écart-type': np.std(auc_scores),
        'Variance': np.var(auc_scores),
        'Coefficient de variation': (np.std(auc_scores) / np.mean(auc_scores)) * 100,
        'Skewness': stats.skew(auc_scores),
        'Kurtosis': stats.kurtosis(auc_scores),
        'IC 95% inf': np.percentile(auc_scores, 2.5),
        'IC 95% sup': np.percentile(auc_scores, 97.5),
        'IC 90% inf': np.percentile(auc_scores, 5),
        'IC 90% sup': np.percentile(auc_scores, 95)
    }

    for key, value in stats_dict.items():
        if 'IC' in key:
            print(f"{key}: {value:.4f}")
        elif key == 'Coefficient de variation':
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.4f}")

    # test de normalité
    _, p_value = stats.normaltest(auc_scores)
    print(f"\nTest de normalité (p-value): {p_value:.4f}")
    if p_value > 0.05:
        print("→ La distribution semble normale")
    else:
        print("→ La distribution ne semble pas normale")

    return stats_dict

# calcul de l'AUC sur test
y_pred_proba_full = logit_result.predict(X_test_sm)
auc_full = roc_auc_score(y_test, y_pred_proba_full)
print(f"AUC sur le test set complet: {auc_full:.4f}")

# application du bootstrap
auc_scores, fig = bootstrap_auc_histogram(
    model=logit_result,
    X_test=X_test_sm,
    y_test=y_test,
    bootstrap_fraction=0.08,
    n_bootstrap=500,
    random_state=42
)

# analyse détaillée
stats_dict = detailed_bootstrap_analysis(auc_scores, "Regression Logistique")

# bootstrap
def bootstrap_auc_histogram(model, X_test, y_test, bootstrap_fraction=0.05,
                           n_bootstrap=1000, random_state=42, figsize=(12, 8)):
    """
    Réalise un bootstrap sur un pourcentage de la base test et affiche l'histogramme des AUC
    """

    print("=" * 60)
    print("BOOTSTRAP AUC ANALYSIS")
    print("=" * 60)

    # calcul taille échantillon bootstrap
    sample_size = int(len(X_test) * bootstrap_fraction)

    print(f"Taille totale du test set: {len(X_test)}")
    print(f"Taille des échantillons bootstrap: {sample_size} ({bootstrap_fraction*100}%)")
    print(f"Nombre d'itérations bootstrap: {n_bootstrap}")

    # stockage des scores AUC
    auc_scores = []

    # boucle bootstrap
    for i in range(n_bootstrap):
        # échnantillonnage avec remise
        X_bootstrap, y_bootstrap = resample(
            X_test, y_test,
            n_samples=sample_size,
            replace=True,
            random_state=random_state + i,
            stratify=y_test
        )

        if not hasattr(model, 'predict_proba'):
        # prédictions
          y_pred_proba = model.predict(X_bootstrap)

        else:
          y_pred_proba = model.predict_proba(X_bootstrap)[:,1]

        auc = roc_auc_score(y_bootstrap,y_pred_proba)

        auc_scores.append(auc)

    # conversion en array
    auc_scores = np.array(auc_scores)

    # calcul des statistiques
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    median_auc = np.median(auc_scores)
    ci_lower = np.percentile(auc_scores, 2.5)
    ci_upper = np.percentile(auc_scores, 97.5)

    # affichage des statistiques
    print(f"\nSTATISTIQUES DES SCORES AUC:")
    print(f"Moyenne: {mean_auc:.4f}")
    print(f"Écart-type: {std_auc:.4f}")
    print(f"Médiane: {median_auc:.4f}")
    print(f"Intervalle de confiance 95%: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Minimum: {np.min(auc_scores):.4f}")
    print(f"Maximum: {np.max(auc_scores):.4f}")

    # création de l'histogramme
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # histogramme principal
    n, bins, patches = ax1.hist(auc_scores, bins=50, alpha=0.7, color='skyblue',
                               edgecolor='black', density=True)

    # lignes verticales pour les statistiques
    ax1.axvline(mean_auc, color='red', linestyle='--', linewidth=2,
                label=f'Moyenne: {mean_auc:.4f}')
    ax1.axvline(median_auc, color='green', linestyle='--', linewidth=2,
                label=f'Médiane: {median_auc:.4f}')
    ax1.axvline(ci_lower, color='orange', linestyle=':', linewidth=1.5,
                label=f'IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax1.axvline(ci_upper, color='orange', linestyle=':', linewidth=1.5)

    ax1.set_xlabel('Score AUC')
    ax1.set_ylabel('Densité')
    ax1.set_title(f'Distribution des scores AUC - Bootstrap {bootstrap_fraction*100}% du test set\n'
                 f'{n_bootstrap} itérations - {sample_size} échantillons par itération')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # box plot
    ax2.boxplot(auc_scores, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Score AUC')
    ax2.set_title('Boxplot des scores AUC')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return auc_scores, fig

# fonction pour analyse détaillée
def detailed_bootstrap_analysis(auc_scores, model_name="Modèle"):
    """
    Analyse détaillée des résultats bootstrap
    """
    print("\n" + "=" * 50)
    print(f"ANALYSE DÉTAILLÉE - {model_name.upper()}")
    print("=" * 50)

    # Statistiques descriptives
    from scipy import stats

    stats_dict = {
        'Moyenne': np.mean(auc_scores),
        'Médiane': np.median(auc_scores),
        'Écart-type': np.std(auc_scores),
        'Variance': np.var(auc_scores),
        'Coefficient de variation': (np.std(auc_scores) / np.mean(auc_scores)) * 100,
        'Skewness': stats.skew(auc_scores),
        'Kurtosis': stats.kurtosis(auc_scores),
        'IC 95% inf': np.percentile(auc_scores, 2.5),
        'IC 95% sup': np.percentile(auc_scores, 97.5),
        'IC 90% inf': np.percentile(auc_scores, 5),
        'IC 90% sup': np.percentile(auc_scores, 95)
    }

    for key, value in stats_dict.items():
        if 'IC' in key:
            print(f"{key}: {value:.4f}")
        elif key == 'Coefficient de variation':
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.4f}")

    # test de normalité
    _, p_value = stats.normaltest(auc_scores)
    print(f"\nTest de normalité (p-value): {p_value:.4f}")
    if p_value > 0.05:
        print("→ La distribution semble normale")
    else:
        print("→ La distribution ne semble pas normale")

    return stats_dict

# calcul de l'AUC sur test
y_pred_proba_full = logit_result.predict(X_test_sm)
auc_full = roc_auc_score(y_test, y_pred_proba_full)
print(f"AUC sur le test set complet: {auc_full:.4f}")

# application du bootstrap
auc_scores, fig = bootstrap_auc_histogram(
    model=logit_result,
    X_test=X_test_sm,
    y_test=y_test,
    bootstrap_fraction=0.08,
    n_bootstrap=500,
    random_state=42
)

# Analyse détaillée
stats_dict = detailed_bootstrap_analysis(auc_scores, "Regression Logistique")

# test de multicollinéarité du VIF (variance inflation factor)
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data.sort_values("VIF", ascending=False)
    return vif_data

# test d'ajustement de Hosmer-Lemeshow
def hosmer_lemeshow_simple(y_true, y_pred_proba, groups=10):
    df = pd.DataFrame({"y": y_true, "prob": y_pred_proba})
    df['groupe'] = pd.qcut(df['prob'], groups, labels=False)

    stats = df.groupby('groupe').agg(
        obs_total=("y", "count"),
        obs_pos=('y', 'sum'),
        prob_sum=('prob', 'sum')
    )

    hl_stat = ((stats['obs_pos'] - stats['prob_sum'])**2 /
               (stats['prob_sum'] * (1 - stats['prob_sum']/stats['obs_total']))).sum()

    p_value = chi2.sf(hl_stat, groups-2)
    return hl_stat, p_value

# fonction random grid search (RGS)
def quick_random_search(model, param_grid, X_val, y_val, n_iter=50, cv=3):
    """
    Random Grid Search
    """

    random_search = RandomizedSearchCV(
        model, param_grid, n_iter=n_iter, cv=cv,
        scoring='roc_auc', random_state=42, n_jobs=-1
    )

    random_search.fit(X_val, y_val)

    print(f"Meilleur score: {random_search.best_score_:.4f}")
    print("Meilleurs paramètres:", random_search.best_params_)

    return random_search.best_estimator_, random_search.best_params_

# calcul du score sur train, test et validation
def score_train_val_test(y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred):
    """ Calcul du score pour train, test et validation

    Paramètres :
    y_train : vrai label du train
    y_train_pred : probabilités prédites pour le train
    y_val : vrai label de validation
    y_val_pred : probabilités prédites pour validation
    y_test : vrai label pour test
    y_test_pred : probabilités prédites pour test

    """

    # vérification prédictions
    for pred, name in zip([y_train_pred, y_val_pred, y_test_pred],
                         ['Train', 'Validation', 'Test']):
        if np.any(pred < 0) or np.any(pred > 1):
            print(f"Attention: Les prédictions pour {name} ne semblent pas être des probabilités")

    # calcul des métriques
    metrics = {}

    # AUC
    try:
        metrics['AUC'] = {
            'Train': roc_auc_score(y_train, y_train_pred),
            'Val': roc_auc_score(y_val, y_val_pred),
            'Test': roc_auc_score(y_test, y_test_pred)
        }
    except Exception as e:
        print(f"Erreur dans le calcul de l'AUC: {e}")
        metrics['AUC'] = {'Train': np.nan, 'Val': np.nan, 'Test': np.nan}

    # log-loss
    try:
        metrics['LogLoss'] = {
            'Train': log_loss(y_train, y_train_pred),
            'Val': log_loss(y_val, y_val_pred),
            'Test': log_loss(y_test, y_test_pred)
        }
    except Exception as e:
        print(f"Erreur dans le calcul de la LogLoss: {e}")
        metrics['LogLoss'] = {'Train': np.nan, 'Val': np.nan, 'Test': np.nan}

    # entropie croisée binaire (identique à log_loss pour la classification binaire)
    try:
        def binary_cross_entropy(y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        metrics["Entropy"] = {
            'Train': binary_cross_entropy(y_train, y_train_pred),
            'Val': binary_cross_entropy(y_val, y_val_pred),
            'Test': binary_cross_entropy(y_test, y_test_pred)
        }
    except Exception as e:
        print(f"Erreur dans le calcul de l'Entropy: {e}")
        metrics["Entropy"] = {'Train': np.nan, 'Val': np.nan, 'Test': np.nan}

    # création du dataframe
    table_of_results = pd.DataFrame(metrics).T
    table_of_results = table_of_results[['Train', 'Val', 'Test']]

    # affichage des résultats
    print("=" * 50)
    print("SCORES SUR LES DIFFÉRENTS ENSEMBLES")
    print("=" * 50)
    print(table_of_results.round(4))
    print("\n" + "=" * 50)

    return table_of_results
