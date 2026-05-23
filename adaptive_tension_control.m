% =========================================================================
% PROJE: Yapay Zeka Destekli Adaptif Gergi Sistemli Tank Paleti
% YAZAR: Ayça Bahar Güner
% ÖZELLİK: Anti-Windup Korumalı ve Kontrol Sinyali Görselleştirmeli PID
% =========================================================================

clear; clc; close all;

%% 1. SİMÜLASYON PARAMETRELERİ
dt = 0.05;              
t_end = 60;             
t = 0:dt:t_end;         
N = length(t);

T_nominal = 50000;      
alpha = 15;             
beta = 25000;           

% PID Katsayıları
Kp = 0.8;               
Ki = 0.15;              
Kd = 0.05;              

% Aktüatör Limitleri
u_max = 15000;
u_min = -15000;

%% 2. SENTETİK SENSÖR VERİSİ (Zemin ve Hız)
v_arac = zeros(1, N);   
mu_zemin = zeros(1, N); 

for i = 1:N
    v_arac(i) = 10 + 5*sin(0.1*t(i)) + randn()*0.5; 
    
    if t(i) < 20
        mu_zemin(i) = 0.8;  % Asfalt
    elseif t(i) < 40
        mu_zemin(i) = 0.3;  % Çamur
    else
        mu_zemin(i) = 0.6;  % Kayalık
    end
    mu_zemin(i) = mu_zemin(i) + randn()*0.02; 
end

%% 3. HEDEF GERGİNLİK (ML Çıktısı Simülasyonu)
T_hedef = zeros(1, N);
for i = 1:N
    T_hedef(i) = T_nominal + alpha*(v_arac(i)^2) - beta*mu_zemin(i);
end

%% 4. ANTI-WINDUP KORUMALI PID DÖNGÜSÜ
T_gercek = zeros(1, N); 
T_gercek(1) = T_nominal;

u_history = zeros(1, N); % Kontrol sinyalini kaydetme dizisi
u_history(1) = 0;        % t=0 anında sistem dengede olduğu için aktüatör valf basıncı 0 kabul edilir

e = zeros(1, N);        
e_int = 0;              

for i = 2:N
    % Hata hesaplama
    e(i) = T_hedef(i) - T_gercek(i-1);
    
    % P ve D Terimleri
    P = Kp * e(i);
    D = Kd * (e(i) - e(i-1)) / dt;
    
    % --- ANTI-WINDUP (CLAMPING) MANTIĞI ---
    u_test = P + (Ki * e_int) + D; 
    
    if (u_test >= u_max && e(i) > 0) || (u_test <= u_min && e(i) < 0)
        % Aktüatör doyuma ulaştı! İntegral donduruldu (Windup önleme).
    else
        % Normal çalışma bölgesi: İntegral güvenle biriktirilebilir.
        e_int = e_int + e(i) * dt; 
    end
    
    I = Ki * e_int;
    u = P + I + D;
    
    % Aktüatör Çıkışını Sınırla (Saturation)
    if u > u_max, u = u_max; end
    if u < u_min, u = u_min; end
    
    u_history(i) = u; % Çizim için kontrol sinyalini kaydet
    
    % Dinamik Güncelleme (Gecikme Modeli)
    tau = 0.5; 
    dT = (u - (T_gercek(i-1) - T_nominal)) / tau;
    T_gercek(i) = T_gercek(i-1) + dT * dt;
end

%% 5. GÖRSELLEŞTİRME
figure('Name', 'Anti-Windup Korumalı Adaptif Gergi Sistemi', 'Position', [100, 50, 900, 850]);

% 1. Grafik: Hız
subplot(4,1,1);
plot(t, v_arac, 'b', 'LineWidth', 1.5);
title('Araç Kinematiği (Sensör Verisi)');
ylabel('Hız (m/s)'); grid on;

% 2. Grafik: Zemin Sürtünmesi
subplot(4,1,2);
plot(t, mu_zemin, 'r', 'LineWidth', 1.5);
title('Zemin Sürtünme Katsayısı (\mu)');
ylabel('\mu Değeri'); grid on;

% 3. Grafik: Gerginlik Takibi
subplot(4,1,3);
plot(t, T_hedef, 'k--', 'LineWidth', 1.8); hold on;
plot(t, T_gercek, 'g', 'LineWidth', 1.5);
title('Anti-Windup Korumalı PID Performansı');
ylabel('Gerginlik (N)'); 
legend('Hedef Gerginlik', 'Gerçek Gerginlik', 'Location', 'Best');
grid on;

% 4. Grafik: Kontrol Sinyali ve Limitler
subplot(4,1,4);
plot(t, u_history, 'm', 'LineWidth', 1.5); hold on;
yline(u_max, 'r--', 'Maks. Valf Basıncı', 'LabelHorizontalAlignment', 'left');
yline(u_min, 'r--', 'Min. Valf Basıncı', 'LabelHorizontalAlignment', 'left');
title('Aktüatör Kontrol Sinyali (u) ve Fiziksel Limitler');
ylabel('Kuvvet (N)'); xlabel('Zaman (s)');
ylim([-18000 18000]);
grid on;
