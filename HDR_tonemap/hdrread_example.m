hdr = hdrread('office.hdr');
hdrwrite(hdr,'office.hdr');
rgb = tonemap(hdr);
imshow(rgb);