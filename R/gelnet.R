## gelnet.R - Generalized Elastic Nets
##
## by Artem Sokolov

#' Linear regression objective function value
#'
#' Evaluates the linear regression objective function value for a given model.
#' See details.
#'
#' Computes the objective function value according to
#' \deqn{ \frac{1}{2n} \sum_i a_i (z_i - (w^T x_i + b))^2 + R(w) }
#'  where
#' \deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
#'
#' @param w p-by-1 vector of model weights
#' @param b the model bias term
#' @param X n-by-p matrix of n samples in p dimensions
#' @param z n-by-1 response vector
#' @param l1 L1-norm penalty scaling factor \eqn{\lambda_1}
#' @param l2 L2-norm penalty scaling factor \eqn{\lambda_2}
#' @param a n-by-1 vector of sample weights
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature-feature penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @return The objective function value.
#' @seealso \code{\link{gelnet.lin}}
gelnet.lin.obj <- function( w, b, X, z, l1, l2, a=rep(1,nrow(X)),
                           d=rep(1,ncol(X)), P=diag(ncol(X)), m=rep(0,ncol(X)) )
  {
    n <- nrow(X)
    p <- ncol(X)

    ## Compute the residual and loss
    r <- z - (X %*% w + b)
    L <- mean( a * r * r) / 2
    R1 <- l1 * t(d) %*% abs(w)
    R2 <- l2 * t(w-m) %*% P %*% (w-m) / 2

    L + R1 + R2
  }

#' Logistic regression objective function value
#'
#' Evaluates the logistic regression objective function value for a given model.
#' See details.
#
#' Computes the objective function value according to
#' \deqn{ -\frac{1}{n} \sum_i y_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
#'  where
#' \deqn{ s_i = w^T x_i + b }
#' \deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
#'
#' @param w p-by-1 vector of model weights
#' @param b the model bias term
#' @param X n-by-p matrix of n samples in p dimensions
#' @param y n-by-1 binary response vector sampled from {0,1}
#' @param l1 L1-norm penalty scaling factor \eqn{\lambda_1}
#' @param l2 L2-norm penalty scaling factor \eqn{\lambda_2}
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature-feature penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @return The objective function value.
#' @seealso \code{\link{gelnet.logreg}}
gelnet.logreg.obj <- function( w, b, X, y, l1, l2, d=rep(1,ncol(X)),
                              P = diag(ncol(X)), m = rep(0,ncol(X)) )
  {
    stopifnot( sort(unique(y)) == c(0,1) )
    s <- X %*% w + b
    R1 <- l1 * t(d) %*% abs(w)
    R2 <- l2 * t(w-m) %*% P %*% (w-m) / 2
    LL <- mean( y * s - log(1+exp(s)) )
    R1 + R2 - LL
  }

#' One-class regression objective function value
#'
#' Evaluates the one-class objective function value for a given model
#' See details.
#'
#' Computes the objective function value according to
#' \deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
#'  where
#' \deqn{ s_i = w^T x_i }
#' \deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
#'
#' @param w p-by-1 vector of model weights
#' @param X n-by-p matrix of n samples in p dimensions
#' @param l1 L1-norm penalty scaling factor \eqn{\lambda_1}
#' @param l2 L2-norm penalty scaling factor \eqn{\lambda_2}
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature-feature penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @return The objective function value.
#' @seealso \code{\link{gelnet.oneclass}}
gelnet.oneclass.obj <- function( w, X, l1, l2, d, P, m )
  {
    s <- X %*% w
    LL <- mean( s - log( 1 + exp(s) ) )
    R1 <- l1 * t(d) %*% abs(w)
    R2 <- l2 * t(w-m) %*% P %*% (w-m) / 2
    R1 + R2 - LL
  }

#' The largest meaningful value of the L1 parameter
#'
#' Computes the smallest value of the LASSO coefficient L1 that leads to an
#'  all-zero weight vector for a given linear regression problem.
#'
#' The cyclic coordinate descent updates the model weight \eqn{w_k} using a soft threshold operator
#' \eqn{ S( \cdot, \lambda_1 d_k ) } that clips the value of the weight to zero, whenever the absolute
#' value of the first argument falls below \eqn{\lambda_1 d_k}. From here, it is straightforward to compute
#' the smallest value of \eqn{\lambda_1}, such that all weights are clipped to zero.
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @param y n-by-1 response vector
#' @param a n-by-1 vector of sample weights
#' @param d p-by-1 vector of feature weights
#' @return The largest meaningful value of the L1 parameter (i.e., the smallest value that yields a model with all zero weights)
L1.ceiling <- function( X, y, a = rep(1,nrow(X)), d = rep(1,ncol(X)) )
  {
    stopifnot( nrow(X) == length(y) )
    b1 <- sum( a*y ) / sum(a)
    xy <- apply( a*X*(y - b1), 2, mean ) / d
    max( abs(xy) )
  }

#' GELnet for linear regression
#'
#' Constructs a GELnet model for linear regression using coordinate descent.
#'
#' The method operates through cyclical coordinate descent.
#' The optimization is terminated after the desired tolerance is achieved, or after a maximum number of iterations.
#' 
#' @param X n-by-p matrix of n samples in p dimensions
#' @param y n-by-1 vector of response values
#' @param l1 coefficient for the L1-norm penalty
#' @param l2 coefficient for the L2-norm penalty
#' @param a n-by-1 vector of sample weights
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature association penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param w.init initial parameter estimate for the weights
#' @param b.init initial parameter estimate for the bias term
#' @param fix.bias set to TRUE to prevent the bias term from being updated (default: FALSE)
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @return A list with two elements:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#'   \item{b}{scalar, bias term for the linear model}
#' }
gelnet.lin <- function( X, y, l1, l2, a = rep(1,n), d = rep(1,p), P = diag(p),
                       m=rep(0,p), max.iter = 100, eps = 1e-5, w.init = rep(0,p),
                       b.init = sum(a*y)/sum(a), fix.bias=FALSE, silent=FALSE )
  {
    n <- nrow(X)
    p <- ncol(X)

    ## Verify argument dimensionality
    stopifnot( length(y) == n )
    stopifnot( length(a) == n )
    stopifnot( length(d) == p )
    stopifnot( all( dim(P) == c(p,p) ) )
    stopifnot( length(m) == p )
    stopifnot( length(w.init) == p )
    stopifnot( length(b.init) == 1 )
    stopifnot( length(l1) == 1 )
    stopifnot( length(l2) == 1 )

    ## Verify name matching (if applicable)
    if( is.null(colnames(X)) == FALSE && is.null(colnames(P)) == FALSE )
      {
        stopifnot( is.null( rownames(P) ) == FALSE )
        stopifnot( all( colnames(X) == rownames(P) ) )
        stopifnot( all( colnames(X) == colnames(P) ) )
      }

    ## Set the initial parameter estimates
    S <- X %*% w.init + b.init
    Pw <- P %*% (w.init - m)

    ## Call the C routine
    res <- .C( "gelnet_base_opt",
              as.double(X), as.double(y), as.double(a), as.double(d),
              as.double(P), as.double(m), as.double(l1), as.double(l2),
              as.double(S), as.double(Pw), as.integer(n), as.integer(p),
              as.integer(max.iter), as.double(eps), as.integer(fix.bias),
              w = as.double( w.init ), b = as.double(b.init), as.integer(silent) )

    res <- res[c("w","b")]
    names( res$w ) <- colnames(X)

    res
  }

#' GELnet for logistic regression
#'
#' Constructs a GELnet model for logistic regression using the Newton method.
#'
#' The method operates by constructing iteratively re-weighted least squares approximations
#' of the log-likelihood loss function and then calling the linear regression routine
#' to solve those approximations. The least squares approximations are obtained via the Taylor series
#' expansion about the current parameter estimates.
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @param y n-by-1 vector of binary response labels
#' @param l1 coefficient for the L1-norm penalty
#' @param l2 coefficient for the L2-norm penalty
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature association penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param w.init initial parameter estimate for the weights
#' @param b.init initial parameter estimate for the bias term
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @return A list with two elements:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#'   \item{b}{scalar, bias term for the linear model}
#' }
#' @seealso \code{\link{gelnet.lin}}
gelnet.logreg <- function( X, y, l1, l2, d = rep(1,p), P = diag(p), m = rep(0,p),
                          max.iter = 100, eps = 1e-5, w.init = rep(0,p),
                          b.init = 0.5, silent = FALSE )
  {
    n <- nrow(X)
    p <- ncol(X)

    ## Verify argument dimensionality
    stopifnot( length( unique(y) ) == 2 )
    stopifnot( length(y) == n )
    stopifnot( length(d) == p )
    stopifnot( all( dim(P) == c(p,p) ) )
    stopifnot( length( w.init ) == p )
    stopifnot( length( b.init ) == 1 )
    stopifnot( length(l1) == 1 )
    stopifnot( length(l2) == 1 )

    ## Verify name matching (if applicable)
    if( is.null(colnames(X)) == FALSE && is.null(colnames(P)) == FALSE )
      {
        stopifnot( is.null( rownames(P) ) == FALSE )
        stopifnot( all( colnames(X) == rownames(P) ) )
        stopifnot( all( colnames(X) == colnames(P) ) )
      }

    ## Convert the labels to {0,1}
    y <- as.integer( y == max(y) )
    
    ## Set the initial parameter estimates
    S <- X %*% w.init + b.init
    Pw <- P %*% (w.init - m)

    res <- .C( "gelnet_logreg_opt",
              as.double(X), as.integer(y), as.double(d), as.double(P),
              as.double(m), as.double(l1), as.double(l2),
              as.double(S), as.double(Pw), as.integer(n), as.integer(p),
              as.integer(max.iter), as.double(eps), w = as.double( w.init ),
              b = as.double(b.init), as.integer(silent) )

    res <- res[c("w","b")]
    names( res$w ) <- colnames(X)
    
    res
  }

#' GELnet for one-class regression
#'
#' Constructs a GELnet model for one-class regression using the Newton method.
#'
#' The function optimizes the following objective:
#' \deqn{ -\frac{1}{n} \sum_i s_i - \log( 1 + \exp(s_i) ) + R(w) }
#'  where
#' \deqn{ s_i = w^T x_i }
#' \deqn{ R(w) = \lambda_1 \sum_j d_j |w_j| + \frac{\lambda_2}{2} (w-m)^T P (w-m) }
#' The method operates by constructing iteratively re-weighted least squares approximations
#' of the log-likelihood loss function and then calling the linear regression routine
#' to solve those approximations. The least squares approximations are obtained via the Taylor series
#' expansion about the current parameter estimates.
#'
#' @param X n-by-p matrix of n samples in p dimensions
#' @param l1 coefficient for the L1-norm penalty
#' @param l2 coefficient for the L2-norm penalty
#' @param d p-by-1 vector of feature weights
#' @param P p-by-p feature association penalty matrix
#' @param m p-by-1 vector of translation coefficients
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param w.init initial parameter estimate for the weights
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @return A list with one element:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#' }
#' @seealso \code{\link{gelnet.lin}}, \code{\link{gelnet.oneclass.obj}}
gelnet.oneclass <- function( X, l1, l2, d = rep(1,p), P = diag(p), m = rep(0,p),
                            max.iter = 100, eps = 1e-5,
                            w.init = rep(0,p), silent= FALSE )
  {
    n <- nrow(X)
    p <- ncol(X)

    ## Verify argument dimensionality
    stopifnot( length(d) == p )
    stopifnot( all( dim(P) == c(p,p) ) )
    stopifnot( length(m) == p )
    stopifnot( length(w.init) == p )
    stopifnot( length(l1) == 1 )
    stopifnot( length(l2) == 1 )

    ## Verify name matching (if applicable)
    if( is.null(colnames(X)) == FALSE && is.null(colnames(P)) == FALSE )
      {
        stopifnot( is.null( rownames(P) ) == FALSE )
        stopifnot( all( colnames(X) == rownames(P) ) )
        stopifnot( all( colnames(X) == colnames(P) ) )
      }

    ## Set the initial parameter estimates
    w <- w.init

    ## Run Newton's method
    fprev <- gelnet.oneclass.obj( w, X, l1, l2, d, P, m )
    for( iter in 1:max.iter )
      {
        if( silent == FALSE )
          {
            cat( "Iteration", iter, ": " )
            cat( "f =", fprev, "\n" )
          }

        ## Compute the current fit
        s <- X %*% w
        pr <- 1 / ( 1 + exp(-s) )

        ## Compute the sample weights and active response
        a <- pr * (1-pr)
        z <- s + 1/pr

        ## Run coordinate descent for the resulting regression problem
        mm <- gelnet.lin( X, z, l1, l2, a, d, P, m, iter*2, eps, w, 0, TRUE, TRUE )
        w <- mm$w

        f <- gelnet.oneclass.obj( w, X, l1, l2, d, P, m )
        if( abs(f - fprev) / abs(fprev) < eps ) break
        else fprev <- f
      }
    
    list( w = w )
  }

#' A GELnet model with a requested number of non-zero weights
#'
#' Binary search to find an L1 penalty parameter value that yields the desired
#'   number of non-zero weights in a GELnet model.
#'
#' The method performs simple binary search starting in [0, l1s] and iteratively
#' training a model using the provided \code{f.gelnet}. At each iteration, the
#' method checks if the number of non-zero weights in the model is higher or lower
#' than the requested \code{nF} and adjusts the value of the L1 penalty term accordingly.
#' For linear regression problems, it is recommended to initialize \code{l1s} to the output
#' of \code{L1.ceiling}.
#' 
#' @param f.gelnet a function that accepts one parameter: L1 penalty value,
#'    and returns a typical GELnets model (list with w and b as its entries)
#' @param nF the desired number of non-zero features
#' @param l1s the right side of the search interval: search will start in [0, l1s]
#' @param max.iter the maximum number of iterations of the binary search
#'
#' @return The model with the desired number of non-zero weights and the corresponding value of the
#' L1-norm parameter. Returned as a list with three elements:
#' \describe{
#'   \item{w}{p-by-1 vector of p model weights}
#'   \item{b}{scalar, bias term for the linear model}
#'   \item{l1}{scalar, the corresponding value of the L1-norm parameter}
#' }
#' @seealso \code{\link{L1.ceiling}}
#' @examples
#' X <- matrix( rnorm(100*20), 100, 20 )
#' y <- rnorm(100)
#' l1s <- L1.ceiling( X, y )
#' f <- function( l1 ) {gelnet.lin( X, y, l1, l2 = 1 )}
#' m <- gelnet.L1bin( f, nF = 50, l1s = l1s )
#' print( m$l1 )
gelnet.L1bin <- function( f.gelnet, nF, l1s, max.iter=10 )
  {
    ## Set up the search region
    L1top <- l1s
    L1bot <- 0

    ## Perform binary search
    for( i in 1:max.iter )
      {
        cat( "Binary search iteration", i, "\n" )
        l1 <- (L1top + L1bot) / 2
        m <- f.gelnet( l1 )
        k <- sum( m$w != 0 )
        cat( "Learned a model with", k, "non-zero features\n" )
        if( k == nF ) break
        if( k < nF ) {L1top <- l1} else {L1bot <- l1}
      }

    ## Store the selected L1 parameter value into the model
    m$l1 <- l1
    m
  }

#' Generate a graph Laplacian
#'
#' Generates a graph Laplacian from the graph adjacency matrix.
#'
#' A graph Laplacian is defined as:
#' \eqn{ l_{i,j} = deg( v_i ) }, if \eqn{ i = j };
#' \eqn{ l_{i,j} = -1 }, if \eqn{ i \neq j } and \eqn{v_i} is adjacent to \eqn{v_j};
#' and \eqn{ l_{i,j} = 0 }, otherwise
#'
#' @param A n-by-n adjacency matrix for a graph with n nodes
#' @return The n-by-n Laplacian matrix of the graph
#' @seealso \code{\link{adj2nlapl}}
adj2lapl <- function( A )
  {
    n <- nrow(A)
    stopifnot( ncol(A) == n )
    
    ## Compute the off-diagonal entries
    L <- -A
    diag(L) <- 0

    ## Compute the diagonal entries
    ## Degree of a node: sum of weights on the edges
    s <- apply( L, 2, sum )
    diag(L) <- -s	## Negative because L == -A
    L
  }

#' Generate a normalized graph Laplacian
#'
#' Generates a normalized graph Laplacian from the graph adjacency matrix.
#'
#' A normalized graph Laplacian is defined as:
#' \eqn{ l_{i,j} = 1 }, if \eqn{ i = j };
#' \eqn{ l_{i,j} = - 1 / \sqrt{ deg(v_i) deg(v_j) } }, if \eqn{ i \neq j } and \eqn{v_i} is adjacent to \eqn{v_j};
#' and \eqn{ l_{i,j} = 0 }, otherwise
#'
#' @param A n-by-n adjacency matrix for a graph with n nodes
#' @return The n-by-n Laplacian matrix of the graph
#' @seealso \code{\link{adj2nlapl}}
adj2nlapl <- function(A)
  {
    n <- nrow(A)
    stopifnot( ncol(A) == n )

    ## Zero out the diagonal
    diag(A) <- 0
    
    ## Degree of a node: sum of weights on the edges
    d <- 1 / apply( A, 2, sum )
    d <- sqrt(d)

    ## Account for "free-floating" nodes
    j <- which( is.infinite(d) )
    d[j] <- 0

    ## Compute the non-normalized Laplacian
    L <- adj2lapl( A )

    ## Normalize
    res <- t( L*d ) * d
    rownames(res) <- rownames(A)
    colnames(res) <- rownames(res)
    res
  }

#' Kernel ridge regression
#'
#' Learns a kernel ridge regression model.
#'
#' The entries in the kernel matrix K can be interpreted as dot products
#' in some feature space \eqn{\phi}. The corresponding weight vector can be
#' retrieved via \eqn{w = \sum_i v_i \phi(x_i)}. However, new samples can be
#' classified without explicit access to the underlying feature space:
#' \deqn{w^T \phi(x) + b = \sum_i v_i \phi^T (x_i) \phi(x) + b = \sum_i v_i K( x_i, x ) + b}
#'
#' @param K n-by-n matrix of pairwise kernel values over a set of n samples
#' @param y n-by-1 vector of response values
#' @param a n-by-1 vector of samples weights
#' @param lambda scalar, regularization parameter
#' @param fix.bias set to TRUE to force the bias term to 0 (default: FALSE)
#' @return A list with two elements:
#' \describe{
#'   \item{v}{n-by-1 vector of kernel weights}
#'   \item{b}{scalar, bias term for the model}
#' }
#' @seealso \code{\link{gelnet.lin}}
gelnet.krr <- function( K, y, a, lambda, fix.bias=FALSE )
  {
    ## Argument verification
    n <- nrow(K)
    stopifnot( length(y) == n )
    stopifnot( length(a) == n )
    stopifnot( is.null(dim(a)) )	## a must not be in matrix form

    ## Set up the sample weights and kernalization of the bias term
    A <- diag(a)

    if( fix.bias )
      {
        ## Solve the optimization problem in closed form
        m <- solve( K %*% A %*% t(K) + lambda * n * K, K %*% A %*% y )
        list( v = m, b = 0 )
      }
    else
      {
        ## Set up kernalization of the bias term
        K1 <- rbind( K, 1 )
        K0 <- cbind( rbind( K, 0 ), 0 )
        
        ## Solve the optimization problem in closed form
        m <- solve( K1 %*% A %*% t(K1) + lambda * n * K0, K1 %*% A %*% y )

        ## Retrieve the weights and the bias term
        list( v = m[1:n], b = m[n+1] )
      }
  }

#' Kernel logistic regression
#'
#' Learns a kernel logistic regression model for a binary classification task
#'
#' The method operates by constructing iteratively re-weighted least squares approximations
#' of the log-likelihood loss function and then calling the kernel ridge regression routine
#' to solve those approximations. The least squares approximations are obtained via the Taylor series
#' expansion about the current parameter estimates.
#'
#' 
#' @param K n-by-n matrix of pairwise kernel values over a set of n samples
#' @param y n-by-1 vector of binary response labels
#' @param lambda scalar, regularization parameter
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param v.init initial parameter estimate for the kernel weights
#' @param b.init initial parameter estimate for the bias term
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @return A list with two elements:
#' \describe{
#'   \item{v}{n-by-1 vector of kernel weights}
#'   \item{b}{scalar, bias term for the model}
#' }
#' @seealso \code{\link{gelnet.krr}}, \code{\link{gelnet.logreg}}
gelnet.klr <- function( K, y, lambda, max.iter = 100, eps = 1e-5,
                       v.init=rep(0,nrow(K)), b.init=0.5, silent=FALSE )
  {
    ## Argument verification
    stopifnot( sort(unique(y)) == c(0,1) )
    stopifnot( nrow(K) == ncol(K) )
    stopifnot( nrow(K) == length(y) )

    ## Objective function to minimize
    fobj <- function( v, b )
      {
        s <- K %*% v + b
        R2 <- lambda * t(v) %*% K %*% v / 2
        LL <- mean( y * s - log(1+exp(s)) )
        R2 - LL
      }

    ## Compute the initial objective value
    v <- v.init
    b <- b.init
    fprev <- fobj( v, b )

    ## Reduce kernel logistic regression to kernel regression
    ##  about the current Taylor method estimates
    for( iter in 1:max.iter )
      {
        if( silent == FALSE )
          {
            cat( "Iteration", iter, ": " )
            cat( "f =", fprev, "\n" )
          }
        
        ## Compute the current fit
        s <- drop(K %*% v + b)
        pr <- 1 / (1 + exp(-s))

        ## Snap probabilities to 0 and 1 to avoid division by small numbers
        j0 <- which( pr < eps )
        j1 <- which( pr > (1-eps) )
        pr[j0] <- 0; pr[j1] <- 1

        ## Compute the sample weights and the response
        a <- pr * (1-pr)
        a[c(j0,j1)] <- eps
        z <- s + (y - pr) / a

        ## Run the coordinate descent for the resulting regression problem
        m <- gelnet.krr( K, z, a, lambda )
        v <- m$v
        b <- m$b

        ## Evaluate the objective function and check convergence criteria
        f <- fobj( v, b )
        if( abs(f - fprev) / abs(fprev) < eps ) break
        else fprev <- f
      }

    list( v = v, b = b )
  }

#' Kernel one-class regression
#'
#' Learns a kernel one-class model for a given kernel matrix
#'
#' The method operates by constructing iteratively re-weighted least squares approximations
#' of the log-likelihood loss function and then calling the kernel ridge regression routine
#' to solve those approximations. The least squares approximations are obtained via the Taylor series
#' expansion about the current parameter estimates.
#'
#' 
#' @param K n-by-n matrix of pairwise kernel values over a set of n samples
#' @param lambda scalar, regularization parameter
#' @param max.iter maximum number of iterations
#' @param eps convergence precision
#' @param v.init initial parameter estimate for the kernel weights
#' @param silent set to TRUE to suppress run-time output to stdout (default: FALSE)
#' @return A list with one element:
#' \describe{
#'   \item{v}{n-by-1 vector of kernel weights}
#' }
#' @seealso \code{\link{gelnet.krr}}, \code{\link{gelnet.logreg}}
gelnet.kor <- function( K, lambda, max.iter = 100, eps = 1e-5,
                       v.init = rep(0,nrow(K)), silent= FALSE )
  {
    ## Argument verification
    stopifnot( nrow(K) == ncol(K) )

    ## Define the objective function to optimize
    fobj <- function( v )
      {
        s <- K %*% v
        LL <- mean( s - log( 1 + exp(s) ) )
        R2 <- lambda * t(v) %*% K %*% v / 2
        R2 - LL
      }

    ## Set the initial parameter estimates
    v <- v.init
    fprev <- fobj( v )

    ## Run Newton's method
    for( iter in 1:max.iter )
      {
        if( silent == FALSE )
          {
            cat( "Iteration", iter, ": " )
            cat( "f =", fprev, "\n" )
          }

        ## Compute the current fit
        s <- drop(K %*% v)
        pr <- 1 / (1 + exp(-s))

        ## Compute the sample weights and active response
        a <- pr * (1-pr)
        z <- s + 1/pr

        ## Solve the resulting regression problem
        mm <- gelnet.krr( K, z, a, lambda, fix.bias=TRUE )
        v <- mm$v

        ## Compute the new objective function value
        f <- fobj( v )
        if( abs(f - fprev) / abs(fprev) < eps ) break
        else fprev <- f
      }

    list( v = v )
  }
