function save_frames(f, savename, options)
% saves 3D field as N frames f(:,:,i) for i = {1,..,N}
%
% - f: f(x,y,i), with x as horizontal axis (column)
% - savename: string, name of saved images, suffixed by '_00i', excluding file extension (default: 'frame')
% - ext: string, file extension (default: 'png')
% - colormap: colormap, matrix of size [N 3] describing colour scale (default: cmap())
% - scale: vector [v1 v2], fix the ends of the colorbar to these values (default: [-M M], M = max(|f|))
% - NaN_col: color of NaN values, [R G B] vector (default: [0.5 0.5 0.5], grey)

arguments
    f (:,:,:) double
    savename char                 = 'frame'
    options.ext char              = 'png'
    options.colormap (:,3) double = cmap()
    options.scale (1,:) double    = max(max(max(abs(f)))) * [-1 1];
    options.NaN_col (1,3) double  = [0.5 0.5 0.5]
end

s = size(f); N = s(3);
d = 1 + floor(log(N) / log(10));

for i = 1:N
    fi = squeeze(f(:,:,i));
    save_image(fi, [savename '_' pad_zeros(i,d)], options.ext, ...
        options.colormap, options.scale, options.NaN_col)
end

end

