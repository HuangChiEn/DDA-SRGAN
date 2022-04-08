function feaCell = Feature_Extraction(irisPolar, maskCell)
    % IV. Feature extraction :
    %==========================================================================
    %feaType = input('Please type the num (compare/0 ; LiborMasekIrisCode/1 ; KhalidIrisCode/2(by default)) for choice method.. \n');
    feaType = 2;
    switch feaType
        case 1
            % Set Path, should import iris_lib/irlib/iris/Normal_encoding/encode
            fea_extor = @LiborMasekIrisCode;  
            % input parameters..Libor
            params = {};
            params.shift_range = [-20:2:20];
            params.method = 'LM';
            params.nscales = 1;
            params.minWaveLength = 18;
            params.mult = 1;
            params.sigmaOnf = 0.5;

        case 2
            fea_extor = @KhalidIrisCode;
            % input parameters..Kahlid
            params = {};
            % Not the upsampling, but decide the Harr-wavelete encode result 
            params.resize_height = 30;  %%30
            params.resize_width = 360;  %%360
            params.sigma_x = 3;
            params.sigma_y = 3;

        otherwise
            error('No such iris code method..');
    end
    
    categ_info = size(irisPolar);
    feaCell = cell(categ_info);
    len = length(irisPolar);
    for idx=1:len
        iris_polar = irisPolar{idx};
        mask_lst = maskCell{idx};
        siz = length(iris_polar);
        fea_lst = cell(1, siz);
        
        for jdx=1:siz
            % Feature extraction.
            fea_lst{jdx} = fea_extor(iris_polar{jdx}, params, mask_lst{jdx});
            
        end
        feaCell{idx} = fea_lst;
    end
    
    fprintf('Feature extraction done..\n\n');
end
