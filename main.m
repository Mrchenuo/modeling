clc;
clear;
% DateStrings = '22/12/2012';
% date = datetime(DateStrings,'InputFormat', 'dd/MM/yyyy')
% % timeStrings = {'5:30:00'};
% % D = duration('11:34', 'InputFormat', 'hh:mm')
% D = duration(12,45,7)
% A = date + D
% 
% D2 = duration(12,46,7)
% A2 = date + D2
% A3 = A2 -A
% ��ȡcsv
% crew_file_id = fopen('�����Ű�Data A-Crew.csv','r');
flight_file_id = fopen('�����Ű�Data A-Flight.csv','r');
% formats = '%s %{dd/MM/yyyy}D %{hh:mm}T %s %{dd/MM/yyyy}D %{hh:mm}T %s %s';
formats = '%s %s %s %s %s %s %s %s';
flight_A = textscan(flight_file_id, formats, 'DateLocale','de_DE', 'Delimiter', ',', 'HeaderLines', 1 );
fclose(flight_file_id);

% ���ڸ�ʽת��
for i=1:length(flight_A{1,2})   
    tmp = datetime(flight_A{1,2}(i),'InputFormat', 'MM/dd/yyyy');
    flight_A{1,2}(i) = {tmp};
end

% ʱ���ʽת��
for i=1:length(flight_A{1,3})   
    tmp = duration(flight_A{1,3}(i), 'InputFormat', 'hh:mm');
    flight_A{1,3}(i) = {tmp};
end
disp(flight_A{1,2});
% filename = 'Dow2004-2010.csv';
% formats = ['%{dd-mm-yy}D',repmat('%f',1,30)];
% fid = fopen(filename,'r');
% title = textscan(fid,repmat('%s',1,31),1,'delimiter',',');
% DowJonesTable = textscan(fid,formats,'delimiter',',','collectoutput',1);
% fclose(fid);

% A = importdata('�����Ű�Data A-Flight.csv');
% B = readtable('�����Ű�Data A-Flight.csv',...
%     'Format',...
%     '%s %{dd MMMM yyyy}D %{HH:mm:ss}T %s %{dd MMMM yyyy}D %{HH:mm:ss}T %s %s');