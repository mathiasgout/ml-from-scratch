# Discriminant Analysis

Soit :
- $Y$ la variable discrète à prédire, à $K \geq 2$ classes.
- $X = (X^1, X^2, \dots, X^p)$, les $p$ variables prédictives.

On suppose que :
- $X = x \mid Y = 1 \sim \mathcal{N}(\mu_{1}, \Sigma_{1})$.
- $X = x \mid Y = 2 \sim \mathcal{N}(\mu_{2}, \Sigma_{2})$.
- $\cdots$
- $X=x \mid Y = K \sim \mathcal{N}(\mu_{K}, \Sigma_{K})$.

avec :
- $x \in \mathbb{R}^{p}$ un invidu.
- $\mu_{0}, \dots, \mu_{K} \in \mathbb{R}^{p}$ des moyennes.
- $\Sigma_{0}, \dots, \Sigma_{m-1} \in \mathbb{R}^{p \times p}$ des matrices de covariance (donc symétriques) non dégénérées (déterminants strictement positifs).

On note, $\forall k \in \{1, \dots, K\}$ et $\forall x \in \mathbb{R}^{p}$ :
$$f_k(x; \mu_k, \Sigma_k) = \mathbb{P}(X = x \mid Y = k) = \frac{1}{(2 \pi)^{p/2} \lvert \Sigma_k \lvert ^{1/2}} \text{exp}(-\frac{1}{2}(x-\mu_k)^{T} \Sigma_k^{-1} (x-\mu_k)).$$

la densité de la loi $\mathcal{N}(\mu_{k}, \Sigma_{k})$.

On note aussi : $$\pi_k = {\mathbb{P}(Y = k)}.$$

Et on sait que :

```math
\begin{align*}
    \mathbb{P}(X = x) 
    &= \sum\limits_{k=1}^{K} \mathbb{P}(Y=k) \mathbb{P}(X=x \mid Y = k). \\[10pt]
    & = \sum\limits_{k=1}^{K} \pi_k f_k(x; \mu_k, \Sigma_k).\\
\end{align*}
```

Et donc, d'après le [**théorème de Bayes**](https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_Bayes), on sait que :

```math
\begin{align*}
    &
    \mathbb{P}(X = x \mid Y = k) = \frac{\mathbb{P}(Y = k \mid X = x) \mathbb{P}(X = x)}{\mathbb{P}(Y = k)}.
    \\[15pt]
    \Longleftrightarrow \hspace{3mm} &
    \mathbb{P}(Y = k \mid X = x) = \frac{\mathbb{P}(X = x \mid Y = k) \mathbb{P}(Y = k)}{\mathbb{P}(X = x)}.
\end{align*}
```

D'où :

```math
\mathbb{P}(Y = k \mid X = x) = \frac{\pi_k f_k(x; \mu_k, \Sigma_k)}{ \sum\limits_{l=1}^{K} \pi_l f_l(x; \mu_l, \Sigma_l)}.
```

Il suffit maintenant d'estimer les paramètres, $\pi_k =  {\mathbb{P}(Y = k)}$, $\mu_k$ et $\Sigma_k$ pour pouvoir calculer un estimateur de $\mathbb{P}(Y = k \mid X = x)$ : 

```math
\hat{\mathbb{P}}(Y = k \mid X = x) = \frac{\hat{\pi}_k f_k(x; \hat{\mu}_k, \hat{\Sigma}_k)}{ \sum\limits_{l=0}^{K} \hat{\pi}_l f_l(x; \hat{\mu}_l, \hat{\Sigma}_l)}.
```

## Linear Discriminant Analysis (LDA)

Dans le cas de la **LDA**, on restreint le modèle à des gaussiennes de même matrice de covariance, c'est à dire que :
$$\Sigma_1 = \Sigma_2 = \dots = \Sigma_{K} = \Sigma.$$

Comme, $\forall k \in \{1, \dots, K\}$, 
- $\pi_k \in \mathbb{R} \Longrightarrow$ $K$ paramètres.
- $\mu_k \in \mathbb{R}^p \Longrightarrow$ $Kp$.
- $\Sigma \in \mathbb{R}^{p \times p} \Longrightarrow$ $1+2+\dots + p = \frac{p(p+1)}{2}$, car $\Sigma$ est symétrique.

Alors, on doit estimer : $K + Kp + \frac{p(p+1)}{2}$ paramètres.

### Estimation des probabilités à priori (priors)

On note $N_k = \sum\limits_{i=1}^{n} I(Y=k)$, on a donc $\sum\limits_{k=1}^{K} N_k = n$.

On estime $\pi_k = \mathbb{P}(Y = k)$ par :
$$\hat{\pi}_k = \frac{N_k}{n}.$$

C'est la proportion de $Y=k$ dans notre notre échantillon.

### Estimation des moyennes (means)

On note :
- $\mu_{k} \in \mathbb{R}^p$.
- $(X_1, \dots, X_n)^T \in \mathbb{R}^{n \times p}$, le jeu de données d'apprentissage.
- $\forall i \in \{1, \dots, n\}$, $X_i \in \mathbb{R}^p$.

Le maximum de vraisemblance et la methode des moments
donnent les mêmes estimateurs :

```math
\hat{\mu}_{k} = \frac{1}{N_k} \sum\limits_{i=1}^{n} X_k \hspace{1mm} I(Y=k).
```

C'est la moyenne empirique quand $Y=k$.

### Estimation de la matrice de covariance (covariance matrix)

[On utilise l’estimateur de la méthode des moments qui est sans biais (contrairement à l’estimateur du maximum de vraisemblance)](https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices). Et on utilise aussi la méthode de la [**pooled variance**](https://en.wikipedia.org/wiki/Pooled_variance) qui est une méthode pour estimer la variance d'un échantillon de plusieurs populations (une population pour chaque classe de $Y$).

On rappelle que :
- $\mu_{k} \in \mathbb{R}^p$.
- $(X_1, \dots, X_n)^T \in \mathbb{R}^{n \times p}$, le jeu de données d'apprentissage.
- $Y \in \{0, \dots, K\}$, $K$ classes.

L'estimateurs de $\Sigma$ est donc :
$$\hat{\Sigma} = \frac{1}{n-K} \sum\limits_{k=1}^{K} \left[ \sum\limits_{i=1}^{n} I(Y=k) \left( X_i - \hat{\mu}_k \right) \left( X_i - \hat{\mu}_k \right)^T \right].$$

Chacune des valeurs de l'estimation de $\mu_k$ est prise en compte dans l'estimateur.

### Estimateur LDA

On a :

```math
\begin{align*}
\hat{\mathbb{P}}(Y = k \mid X = x) 
&= \frac{\hat{\pi}_k f_k(x; \hat{\mu}_k, \hat{\Sigma}_k)}{ \sum\limits_{l=1}^{K} \hat{\pi}_l f_l(x; \hat{\mu}_l, \hat{\Sigma}_l)}. \\
&= \frac{\frac{\hat{\pi}_k}{(2 \pi)^{p/2} \lvert \hat{\Sigma} \lvert ^{1/2}} \text{exp}(-\frac{1}{2}(x-\hat{\mu}_k)^{T} \hat{\Sigma}^{-1} (x-\hat{\mu}_k))}{\sum\limits_{l=1}^{K} \frac{\hat{\pi}_l}{(2 \pi)^{p/2} \lvert \hat{\Sigma} \lvert ^{1/2}} \text{exp}(-\frac{1}{2}(x-\hat{\mu}_l)^{T} \hat{\Sigma}^{-1} (x-\hat{\mu}_l))}.
\end{align*}
```

Et donc, notre estimateur **LDA** de $P( Y=k \mid X=x)$ est :
```math
\hat{\mathbb{P}}(Y = k \mid X = x)  = \frac{\hat{\pi}_k \hspace{1mm} \text{exp}(-\frac{1}{2}(x-\hat{\mu}_k)^{T} \hat{\Sigma}^{-1} (x-\hat{\mu}_k))}{\sum\limits_{l=1}^{K} \hat{\pi}_l \hspace{1mm} \text{exp}(-\frac{1}{2}(x-\hat{\mu}_l)^{T} \hat{\Sigma}^{-1} (x-\hat{\mu}_l))}.
```

On peut aussi montrer (je ne sais pas faire la démonstration) que :
$$
log \hspace{1mm} \hat{\mathbb{P}}(Y = k \mid X = x) = x^T \hat{\Sigma}^{-1} \hat{\mu}_k - \frac{1}{2} \hat{\mu}_k \hat{\Sigma}^{-1} \hat{\mu}_k + log \hat{\pi}_k + cst.
$$

où $cst$ est un terme constant.
