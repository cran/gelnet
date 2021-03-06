% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gelnet.R
\name{gelnet.oneclass.obj}
\alias{gelnet.oneclass.obj}
\title{One-class regression objective function value}
\usage{
gelnet.oneclass.obj(w, X, l1, l2, d = rep(1, ncol(X)), P = diag(ncol(X)),
  m = rep(0, ncol(X)))
}
\arguments{
\item{w}{p-by-1 vector of model weights}

\item{X}{n-by-p matrix of n samples in p dimensions}

\item{l1}{L1-norm penalty scaling factor \eqn{\lambda_1}}

\item{l2}{L2-norm penalty scaling factor \eqn{\lambda_2}}

\item{d}{p-by-1 vector of feature weights}

\item{P}{p-by-p feature-feature penalty matrix}

\item{m}{p-by-1 vector of translation coefficients}
}
\value{
The objective function value.
}
\description{
Evaluates the one-class objective function value for a given model
See details.
}
\details{
Computes the objective function value according to
\deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
 where
\deqn{ s_i = w^T x_i }
\deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
}
\seealso{
\code{\link{gelnet}}
}

