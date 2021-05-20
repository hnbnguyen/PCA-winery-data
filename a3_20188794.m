function a3_20188794
% Function for CISC271, Winter 2021, Assignment #3

    % % Read the test data from a CSV file
    % % Extract the data matrix and the Y cluster identifiers
    filename  = 'wine.csv';
    file = csvread(filename, 0, 1)';
    yvec = file(:, 1);
    Xmat = file(:, 2:14);
    
    % % Compute the pair of columns of Xmat with the lowest DB index
    [datarow, datacol] = size(Xmat);
    %creating a matrix, each cell represent a dbindex value for a pair
    dbIndex = zeros(datacol, datacol);
    
    % looping through all possible pairing of columns to find min dbindex
    for idx = 1 : datacol
        for jdx = 1 : datacol
            testMat = [Xmat(:, idx) Xmat(:, jdx)];
            dbScore = dbindex(testMat, yvec);
            % append the correlating score to the cell in the matrix
            dbIndex(idx, jdx) = dbScore;
        end
    end
    
    %finding the minimum value in the matrix
    minimum = min(min(dbIndex));
    %since its a symmetric matrix, min value repeats twice
    [Xmin,Ymin] = find(dbIndex==minimum);
    
    %displaying and plotting result for figure 1
    disp("Lowest DB Index: ")
    disp(minimum)
    disp("Column pair with min DB index: ")
    disp(Ymin)
    
    f1 = figure;
    gscatter(Xmat(:, Ymin(1)),Xmat(:, Ymin(2)), yvec);
    title("Clustering Pair of Values with Lowest DBIndex")
    xlabel("Ethanol")
    ylabel("Flavanoids")
    
    % % Compute the PCA's of the data using the SVD; score the clusterings
    % mean of each col in Xmat
    %creating a matrix containing the mean of each column for running
    %through all rows
    means = ones(length(Xmat), 1) * mean(Xmat, 1); 
    %creating zero mean matrix
    zeroMean = Xmat - means;    
    
    [U E V] = svd(zeroMean, 0);
    %computing the two score vectors and their dbindex value
    score = zeroMean * V(:, [1 2]);
    reducedScore = dbindex(score, yvec);
    
    % plot and display result for figure 2
    disp("Raw PCA Score")
    disp(reducedScore)
   
    f2 = figure;
    gscatter(score(:, 1),score(:, 2), yvec);
    title("PCA with Unstandardized Data")
    xlabel("Z1")
    ylabel("Z2")
    
    % % Compute the Standardized PCA's of the data using the SVD
    % % score the clusterings
    
    % nomarlize the data
    standard = normalize(Xmat);
    %creating the zero mean matrix
    zeroMean_std  = standard - ones(length(standard), 1) * mean(standard, 1);
    
    [U E V] = svd(zeroMean_std, 0);
    
    Zscore = zeroMean_std * V(:, [1,2]);
    yk3 = kmeans(Zscore, 3, 'start', [1 1 ; 0 -2 ; 3 1]);
    sum = 0;
    for idx = 1 : size(yvec)
        if yk3(idx) ~= yvec(idx)
            sum = sum + 1;
        end
    end 
    sum
    
    disp("Standardized PCA score: ")
    db_std = dbindex(Zscore,yvec);
    disp(db_std)
    
    f3 = figure;
    gscatter(Zscore(:,1), Zscore(:, 2), yvec)
    title("PCA with Standardized Data")
    xlabel("Z1")
    ylabel("Z2")
    
    
end
function score = dbindex(Xmat, lvec)
% SCORE=DBINDEX(XMAT,LVEC) computes the Davies-Bouldin index
% for a design matrix XMAT by using the values in LVEC as labels.
% The calculation implements a formula in their journal article.
%
% INPUTS:
%        XMAT  - MxN design matrix, each row is an observation and
%                each column is a variable
%        LVEC  - Mx1 label vector, each entry is an observation label
% OUTPUT:
%        SCORE - non-negative scalar, smaller is "better" separation

    % Anonymous function for Euclidean norm of observations
    rownorm = @(xmat) sqrt(sum(xmat.^2, 2));

    % Problem: unique labels and how many there are
    kset = unique(lvec);
    k = length(kset);

    % Loop over all indexes and accumulate the DB score of each cluster
    % gi is the cluster centroid
    % mi is the mean distance from the centroid
    % Di contains the distance ratios between IX and each other cluster
    D = [];
    for ix = 1:k
        Xi = Xmat(lvec==kset(ix), :);
        gi = mean(Xi);
        mi = mean(rownorm(Xi - gi));
        Di = [];
        for jx = 1:k
            if jx~=ix
                Xj = Xmat(lvec==kset(jx), :);
                gj = mean(Xj);
                mj = mean(rownorm(Xj - gj));
                Di(end+1) = (mi + mj)/norm(gi - gj);
            end
        end
        D(end+1) = max(Di);
    end

    % DB score is the mean of the scores of the clusters
    score = mean(D);
end
