# Discriminant Analysis

Soit :
- $x \in \mathbb{R}^{p}$, un individu à $p$ variables prédictive.
- $y \in \{0, \dots, m-1\}$, la variable discrète à prédire, à $m$ modalités ($m$ classes).

On suppose :
- $X = x \mid y = 0 \sim \mathcal{N}(\mu_{0}, \Sigma_{0})$.
- $X = x \mid y = 1 \sim \mathcal{N}(\mu_{1}, \Sigma_{1})$.
- $\cdots$
- $X=x \mid y = m-1 \sim \mathcal{N}(\mu_{m-1}, \Sigma_{m-1})$.

avec $\mu_{0}, \dots, \mu_{m-1} \in \mathbb{R}^{p}$ et $\Sigma_{0}, \dots, \Sigma_{m-1} \in \mathbb{R}^{p \times p}$ des matrices de covariance (donc symétriques) non dégénérées (déterminants strictement positifs).

On note, $\forall c \in \{0, \dots, m-1\}$ et $\forall x \in \mathbb{R}^{p}$ :
$$f_c(x; \mu_c, \Sigma_c) = \mathbb{P}(X = x \mid y = c) = \frac{1}{(2 \pi)^{p/2} \lvert \Sigma_c \lvert ^{1/2}} \text{exp}(-\frac{1}{2}(x-\mu_c)^{T} \Sigma_c^{-1} (x-\mu_c)).$$

la densité de la loi $\mathcal{N}(\mu_{c}, \Sigma_{c})$.

On note aussi : $$\pi_c = {\mathbb{P}(y = c)}.$$

Et on sait que :

$$
\begin{align*}
    \mathbb{P}(X = x) 
    &= \sum\limits_{k=0}^{m-1} \mathbb{P}(y=k) \mathbb{P}(X=x \mid y = k). \\[10pt]
    & = \sum\limits_{k=0}^{m-1} \pi_k f_k(x; \mu_k, \Sigma_k).\\
\end{align*}
$$

Et donc, d'après le [**théorème de Bayes**](https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_Bayes), on sait que :

$$
\begin{align*}
    &
    \mathbb{P}(X = x \mid y = c) = \frac{\mathbb{P}(y = c \mid X = x) \mathbb{P}(X = x)}{\mathbb{P}(y = c)}.
    \\[15pt]
    \Longleftrightarrow \hspace{3mm} &
    \mathbb{P}(y = c \mid X = x) = \frac{\mathbb{P}(X = x \mid y = c) \mathbb{P}(y = c)}{\mathbb{P}(X = x)}.
\end{align*}
$$

D'où :
$$\mathbb{P}(y = c \mid X = x) = \frac{\pi_c f_c(x; \mu_c, \Sigma_c)}{ \sum\limits_{k=0}^{m-1} \pi_k f_k(x; \mu_k, \Sigma_k)}.$$  

Il suffit maintenant d'estimer les paramètres, $\pi_c =  {\mathbb{P}(y = c)}$, $\mu_c$ et $\Sigma_c$ pour pouvoir calculer un estimateur de $\mathbb{P}(y = c \mid X = x)$ : 
$$
\hat{\mathbb{P}}(y = c \mid X = x) = \frac{\hat{\pi}_c f_c(x; \hat{\mu}_c, \hat{\Sigma}_c)}{ \sum\limits_{k=0}^{m-1} \hat{\pi}_k f_k(x; \hat{\mu}_k, \hat{\Sigma}_k)}.
$$

## Linear Discriminant Analysis (LDA)

Dans le cas de la **LDA**, on restreint le modèle à des gaussiennes de même matrice de covariance, c'est à dire que :
$$\Sigma_0 = \Sigma_1 = \dots = \Sigma_{m-1} = \Sigma.$$

Comme, $\forall c \in \{0, \dots, m-1\}$, 
- $\pi_c \in \mathbb{R} \Longrightarrow$ $m-1$ paramètres (car $\sum\limits_{k=0}^{m-1} \pi_k = n$, donc pas besoin d'estimer le dernier).
- $\mu_c \in \mathbb{R}^p \Longrightarrow$ $mp$.
- $\Sigma \in \mathbb{R}^{p \times p} \Longrightarrow$ $1+2+\dots + p = \frac{p(p+1)}{2}$, car $\Sigma$ est symétrique.

Alors, on doit estimer : $(m-1) + mp + \frac{p(p+1)}{2}$ paramètres.

### Estimation des $\pi_c$ (priors)

On note $N_c = \sum\limits_{i=1}^{n} \mathbb{1}_{y=c}$, on a donc $\sum\limits_{c=0}^{m-1} N_c = n$.

On estime $\pi_c = \mathbb{P}(y = c)$ par :
$$\hat{\pi}_c = \frac{N_c}{n}.$$

C'est la proportion de $y=c$ dans notre notre échantillon.

### Estimation des $\mu_c$ (means)

On note :
- $\mu_{c} \in \mathbb{R}^p$.
- $(X_1, \dots, X_n)^T \in \mathbb{R}^{n \times p}$, le jeu de données d'apprentissage.
- $\forall i \in \{1, \dots, n\}$, $X_i \in \mathbb{R}^p$.

Le maximum de vraisemblance et la methode des moments
donnent les mêmes estimateurs :

$$\hat{\mu}_{c} = \frac{1}{N_c} \sum\limits_{i=1}^{n} X_i \mathbb{1}_{y=c}.$$

C'est la moyenne empirique quand $y=c$.

### Estimation de $\Sigma$ (covariance matrix)

[On utilise l’estimateur de la méthode des moments qui est sans biais (contrairement à l’estimateur du maximum de vraisemblance)](https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices). Et on utilise aussi la méthode de la [**pooled variance**](https://en.wikipedia.org/wiki/Pooled_variance) qui est une méthode pour estimer la variance d'un échantillon de plusieurs populations (une population pour chaque valeur de $y$).

On rappelle que :
- $\mu_{c} \in \mathbb{R}^p$.
- $(X_1, \dots, X_n)^T \in \mathbb{R}^{n \times p}$, le jeu de données d'apprentissage.
- $y \in \{0, \dots, m-1\}$, $m$ classes.

L'estimateurs de $\Sigma$ est donc :

$$\hat{\Sigma} = \frac{1}{n-m} \sum\limits_{c=0}^{m-1} \left[ \sum\limits_{i=1}^{n} \mathbb{1}_{y=c} \left( X_i - \hat{\mu}_c \right) \left( X_i - \hat{\mu}_c \right)^T \right].$$

Chacune des valeurs de l'estimation de $\mu_c$ est prise en compte dans l'estimateur.

### Estimateur LDA de $P( y=1 \mid X=x)$

On a :

$$
\begin{align*}
\hat{\mathbb{P}}(y = c \mid X = x) 
&= \frac{\hat{\pi}_c f_c(x; \hat{\mu}_c, \hat{\Sigma}_c)}{ \sum\limits_{k=0}^{m-1} \hat{\pi}_k f_k(x; \hat{\mu}_k, \hat{\Sigma}_k)}. \\
&= \frac{\frac{\hat{\pi}_c}{(2 \pi)^{p/2} \lvert \hat{\Sigma} \lvert ^{1/2}} \text{exp}(-\frac{1}{2}(x-\hat{\mu}_c)^{T} \hat{\Sigma}^{-1} (x-\hat{\mu}_c))}{\sum\limits_{k=0}^{m-1} \frac{\hat{\pi}_k}{(2 \pi)^{p/2} \lvert \hat{\Sigma} \lvert ^{1/2}} \text{exp}(-\frac{1}{2}(x-\hat{\mu}_k)^{T} \hat{\Sigma}^{-1} (x-\hat{\mu}_k))}.
\end{align*}
$$

Et donc, notre estimateur **LDA** de $P( y=c \mid X=x)$ est :
$$\hat{\mathbb{P}}(y = c \mid X = x)  = \frac{\hat{\pi}_c \hspace{1mm} \text{exp}(-\frac{1}{2}(x-\hat{\mu}_c)^{T} \hat{\Sigma}^{-1} (x-\hat{\mu}_c))}{\sum\limits_{k=0}^{m-1} \hat{\pi}_k \hspace{1mm} \text{exp}(-\frac{1}{2}(x-\hat{\mu}_k)^{T} \hat{\Sigma}^{-1} (x-\hat{\mu}_k))}.$$