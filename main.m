clc;
clear;
DateStrings = '22/12/2012';
date = datetime(DateStrings,'InputFormat', 'dd/MM/yyyy')
% timeStrings = {'5:30:00'};
% D = duration('11:34', 'InputFormat', 'hh:mm')
D = duration(12,45,7)
A = date + D

D2 = duration(12,46,7)
A2 = date + D2
A3 = A2 -A
% ��ȡcsv
file_id = fopen('�����Ű�Data A-Flight.csv','r');
% formats = '%s %{dd/MM/yyyy}D %{hh:mm}T %s %{dd/MM/yyyy}D %{hh:mm}T %s %s';
formats = '%s %s %s %s %s %s %s %s';
crew_A = textscan(file_id, formats, 'DateLocale','de_DE', 'Delimiter', ',', 'HeaderLines', 1 );
fclose(file_id);

% ��ʽת��


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
a = [];  %����null��Ҳ����ʲô������
for i=1:10
  a = [a i];  
end