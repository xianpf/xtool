
    def prob_histgram(pixel_probs_k):
        from io import BytesIO
        from PIL import Image
        buffer = BytesIO()
        _ = plt.hist(pixel_probs_k.cpu().detach().view(-1).numpy(), 100)
        plt.savefig(buffer, format='png')
        plt.clf()
        buffer.seek(0)
        im = np.array(Image.open(buffer))
        return im
