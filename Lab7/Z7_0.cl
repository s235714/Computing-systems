__kernel void conv_corr(sampler_t sampler,read_only image2d_t src_image, write_only image2d_t dst_image)
  {
  size_t global_size_0 = get_global_size(0);
  size_t global_size_1 = get_global_size(1);
  size_t global_id_0 = get_global_id(0);
  size_t global_id_1 = get_global_id(1);
  int2 image_dim = get_image_dim(src_image);

  float kernel_values[3][3] = {{-1, -1, -1},
                        {-1, 8, -1},
                        {-1, -1, -1}};

  float4 pixel = (float4)(0, 0, 0, 0);

  for(int i = -1; i <= 1; ++i){
    for(int j = -1; j <= 1; ++j){
        int2 pos_of_img = (int2)(global_id_0 + i, global_id_1 + j);
        float kv = kernel_values[i+1][j+1];
        float4 v_values = (float4)(kv, kv, kv, kv);
        pixel += read_imagef(src_image, sampler, pos_of_img)*v_values;
    }
  }
  int2 coord = (int2)(global_id_0, global_id_1);
  write_imagef(dst_image, coord, pixel);
}
