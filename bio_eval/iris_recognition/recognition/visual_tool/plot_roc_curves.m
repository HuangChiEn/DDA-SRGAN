function plot_roc_curves(load_cell, legnd_str_lst, plot_params, plot_type)
    fig = figure;
    for idx=1:length(load_cell)
        load(load_cell{idx});
        
        if plot_type == "iris"
            plotSemilogRocCurve(results, plot_params{idx}, [-4 0]);
        else
            tp_fp_strt.tpr = tpr;
            tp_fp_strt.fpr = fpr;
            % Note : the plotRocCurve have been modified for suitable the face recognition !!
            plotRocCurve(plot_params{idx}, tp_fp_strt); 
        end
        
        hold on;
    end
    legend(legnd_str_lst);
    saveas(fig, './result_fig/ROC_fig.bmp');
end