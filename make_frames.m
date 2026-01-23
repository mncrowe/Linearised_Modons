addpath matlab/

delete_frames = true;

files = dir('data/*_window.nc');

for i = 1:length(files)

    filename = files(i).name;
    dirname = ['frames/Frames_' filename(1:end-10)];

    disp(['Creating frames and movie for case ' filename(1:end-10) '...'])

    if exist(dirname) == 7
        rmdir(dirname, 's')
    end
        
    mkdir(dirname)

    q = ncread(['data/' filename], "q");
    save_frames(q, [dirname '/frame'])

    s = size(q, 3);

    n = num2str(strlength(string(size(q, 3))));

    movname = ['mov/mov_' filename(1:end-10) '.mp4'];

    system(['ffmpeg/bin/ffmpeg -y -framerate 20 -i ' dirname '/frame_%0' n 'd.png ' movname ' >/dev/null 2>&1']);
    
    if delete_frames; rmdir(dirname, 's'); end

end

close all
