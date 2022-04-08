function maskCell = Mask_Generation(irisPolar, msk_bnd)
    % III. Mask generation
    %==========================================================================
    %@ 1.Get mask generator.
    %maskType = input('choice mask type, 1 -> Rule based ; 2 -> Gaussian Mixture Models\n');
    maskType = 3;
    categ_info = size(irisPolar);
    
    %@ 2.Generate mask for polar coordinate images.
    maskCell = cell(categ_info);
    len = length(irisPolar);
    for idx=1:len
        polar_iris = irisPolar{idx};
        siz = length(irisPolar{idx});
        msk_lst = cell(1, siz);
        
        bnd = msk_bnd{idx};
        for jdx=1:siz
            switch maskType
                case 1
                   msk_lst{jdx} = createRuleBasedMask(polar_iris{jdx});
                case 2
                    load('./load/GmmModel_7Gb.mat');      % load pre-training model 
                    params.bayesS = bayesS;
                    params.GbPar = GbPar;
                    msk_lst{jdx} = IrisMaskEstiGaborGMM(polar_iris{jdx}, params);
                case 3
                    msk_lst{jdx} = bnd{jdx}; 
                    %si(msk_lst{jdx});
                    %si(polar_iris{jdx});
                    %pause;
                    %close all;
                otherwise
                    error('No such type mask generator..');
            end
 
        end
        maskCell{idx} = msk_lst;
    end
        
    fprintf('Mask generation done..\n\n');
end