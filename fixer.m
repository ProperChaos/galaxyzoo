c = cell(79975, 1);

for i = 1:79975
	c{i} = filterData(i).name(1:6);
end