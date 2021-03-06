% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gelnet.R
\name{gelnet.logreg.obj}
\alias{gelnet.logreg.obj}
\title{Logistic regression objective function value}
\usage{
gelnet.logreg.obj(w, b, X, y, l1, l2, d = rep(1, ncol(X)),
  P = diag(ncol(X)), m = rep(0, ncol(X)), balanced = FALSE)
}
\arguments{
\item{w}{p-by-1 vector of model weights}

\item{b}{the model bias term}

\item{X}{n-by-p matrix of n samples in p dimensions}

\item{y}{n-by-1 binary response vector sampled from {0,1}}

\item{l1}{L1-norm penalty scaling factor \eqn{\lambda_1}}

\item{l2}{L2-norm penalty scaling factor \eqn{\lambda_2}}

\item{d}{p-by-1 vector of feature weights}

\item{P}{p-by-p feature-feature penalty matrix}

\item{m}{p-by-1 vector of translation coefficients}

\item{balanced}{boolean specifying whether the balanced model is being evaluated}
}
\value{
The objective function value.
}
\description{
Evaluates the logistic regression objective function value for a given model.
See details.
Computes the objective function value according to
\deqn{ -\frac{1}{n} \sum_i y_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
 where
\deqn{ s_i = w^T x_i + b }
\deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
When balanced is TRUE, the loss average over the entire data is replaced with averaging
over each class separately. The total loss is then computes as the mean over those
per-class estimates.
}
\seealso{
\code{\link{gelnet}}
}

