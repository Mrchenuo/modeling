%main.m
clc;
clear;
warning('off');

%Parameter settings
num=3;  %Number of fueling points
prob=0.1; %Refueling failure probability
population_size=2000;  %Population size
chromosome_size=10*num;  %Chromosome length
generation_size=30;  %The maximum number of iterations
cross_rate=0.6;  %Cross probability
mutate_rate=0.01;  %Mutation probability
best_fitness=0;


%Individual initialization
for i=1:population_size
    for j=1:chromosome_size
        population(i,j)=round(rand);
    end
end

for G=1:generation_size  %Iteration

%Calculating individual fitness
for i=1:population_size
    fitness_value(i)=0;
end
for i=1:population_size
    fitness_value(i)=fitness(population(i,:),chromosome_size,prob,num);
end
    
%Select
for i=1:population_size
    fitness_sum(i)=0;
end
min_index=1;
temp=1;
temp_chromosome(chromosome_size)=0;
for i=1:population_size
    min_index=i;
    for j=i+1:population_size
        if fitness_value(j)<fitness_value(min_index)
            min_index=j;
        end
    end
    if min_index~=i
        temp=fitness_value(i);
        fitness_value(i)=fitness_value(min_index);
        fitness_value(min_index)=temp;
        for k = 1:chromosome_size
            temp_chromosome(k) = population(i,k);
            population(i,k) = population(min_index,k);
            population(min_index,k) = temp_chromosome(k);
        end
    end
end
for i=1:population_size
    if i==1
        fitness_sum(i) = fitness_sum(i) + fitness_value(i);    
   else
        fitness_sum(i) = fitness_sum(i-1) + fitness_value(i);
    end
end
fitness_average(G) = fitness_sum(population_size)/population_size;
if fitness_value(population_size) > best_fitness
    best_fitness = fitness_value(population_size);
    best_generation = G;
    for j=1:chromosome_size
        best_individual(j) = population(population_size,j);
    end
end

%Chromosome crossing
for i=1:population_size
    r = rand * fitness_sum(population_size); 
    first = 1;
    last = population_size;
    mid = round((last+first)/2);
    idx = -1; 
    while (first <= last) && (idx == -1) 
        if r > fitness_sum(mid)
            first = mid;
        elseif r < fitness_sum(mid)
            last = mid;     
        else
            idx = mid;
            break;
        end
        mid = round((last+first)/2);
        if (last - first) == 1
            idx = last;
            break;
        end
    end
    for j=1:chromosome_size
        population_new(i,j) = population(idx,j);
    end
end
for i=1:population_size
    for j=1:chromosome_size
        population(i,j) = population_new(i,j);
    end
end
for i=1:2:population_size-1
    if(rand < cross_rate)
        cross_position = round(rand * chromosome_size);
        if (cross_position == 0 || cross_position == 1)
            continue;
        end 
        for j=cross_position:chromosome_size
            temp = population(i,j);
            population(i,j) = population(i+1,j);
            population(i+1,j) = temp;
        end
    end
end

%Chromosomal variation
for i=1:population_size
    if rand < mutate_rate
        mutate_position = round(rand*chromosome_size);
        if mutate_position == 0 
            continue;
        end
        population(i,mutate_position) = 1 - population(i, mutate_position);
    end
end
end

%Decode
for i=1:10:chromosome_size
    location(ceil(i/10))=best_individual(i)*512+best_individual(i+1)*256+best_individual(i+2)*128+best_individual(i+3)*64+best_individual(i+4)*32+best_individual(i+5)*16+best_individual(i+6)*8+best_individual(i+7)*4+best_individual(i+8)*2+best_individual(i+9);
end
for i=1:length(location)
    if location(i)>=652
        location(i)=location(i)+206;
    end
end

%Output the optimal solution
disp(location);






%useness.m
%To test whether the program is feasible.
%Returns 1 if the program is feasible. Otherwise, return 0.
function y=useness(loca)
oil=155;
for i=1:length(loca)
    if loca(i)>372&&loca(i)<858
        y=0;break;
    else
        if i>1
            len=loca(i)-loca(i-1);
        else
            len=loca(i);
        end
        oil_use=155/680*len;
        oil=oil-oil_use;
        if oil<0
            y=0;break;
        end
        if loca(i)<=372
            oil_add=170-155/680*loca(i)*2;
        else
            oil_add=170-155/680*(1230-loca(i))*2;
        end
        oil=oil+oil_add;
        if oil>155
            oil=155;
        end
    end
    y=1;
end
if y==1
    len=1230-loca(length(loca));
    oil_use=155/680*len;
    oil=oil-oil_use;
    if oil<0
        y=0;
    else
        y=1;
    end
end
end





%fitness.m
%Calculating the individual fitness
function y=fitness(populate,chromosome_size,prob,num)
for i=1:10:chromosome_size
    loca(ceil(i/10))=populate(i)*512+populate(i+1)*256+populate(i+2)*128+populate(i+3)*64+populate(i+4)*32+populate(i+5)*16+populate(i+6)*8+populate(i+7)*4+populate(i+8)*2+populate(i+9);
end
for i=1:length(loca)
    if loca(i)>=652
        loca(i)=loca(i)+206;
    end
end
loca=sort(loca);
y=useness(loca);
if y ~=0
    y=0;   
    for i=1:num
        index_loca=combntns(loca,i);
        [index_x,~]=size(index_loca);
        for j=1:index_x
            fit=useness(index_loca(j,:));
            if fit==0
                y=y+((1-prob)^i)*(prob^(num-i));
            end
        end
    end
    y=1-y-prob^num;
end















