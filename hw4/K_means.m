function center = K_means(K, data)
trainNum = size(data,1);

%init k centers
for k = 1 : K
    center(k,:) = data(k,1:2);
end

run_flag = true;
while(run_flag)
    %cluster data
    cluster = [];
    cluster_count(1:K) = 0;
    for i = 1 : trainNum
        minD = 10000000;
        for k = 1 : K
            diff = data(i,:) - center(k,:);
            diff = diff.^2;
            dist = sum(diff);

            if( dist < minD )
                idx = k;
                minD = dist;
            end
        end    
        cluster = [cluster idx];
        cluster_count(idx) = cluster_count(idx) + 1;
    end

    %get new centers
    newcenter(1:K,1:2) = 0;
    for i = 1 : trainNum
        idx = cluster(i);
        newcenter(idx,:) = newcenter(idx,:) + data(i,:);
    end
    for k = 1 : K
        newcenter(k,:) = newcenter(k,:)/cluster_count(k);
    end
    
%    disp(center);
%    disp('-----');
%    disp(newcenter);
 
    %check stop criteria
    run_flag = false;
    for k = 1 : K
        if( sum((newcenter(k,:)-center(k,:)).^2) > 0.0001 )
            run_flag = true;
            break;
        end
    end    
    center = newcenter;
end
