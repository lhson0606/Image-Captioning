import torch.backends.cudnn as cudnn
import config.gconfig as gconfig
import torch.optim
import json
from core.datasets import CaptionDataset
import torchvision.transforms as transforms
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from core.models import Encoder, DecoderWithAttention

# Allowlist custom classes
torch.serialization.add_safe_globals([Encoder, DecoderWithAttention])

checkpoint = torch.load("../" + gconfig.checkpoint, weights_only=False, map_location=gconfig.device)
decoder = checkpoint['decoder']
decoder = decoder.to(gconfig.device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(gconfig.device)
encoder.eval()

with open(gconfig.word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluate the model using BLEU-4 score.
    :param beam_size: Beam size
    :return: BLEU-4 score
    """
    loader = torch.utils.data.DataLoader(
        CaptionDataset(gconfig.data_dir, 'coco_5_cap_per_img_5_min_word_freq', 'TEST',
                       transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True
    )

    k = beam_size

    references = list()
    hypotheses = list()

    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="Evaluating" + str(beam_size))):
        image = image.to(gconfig.device)

        encoder_out = encoder(image)
        encoder_dim = encoder_out.size(3)

        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(gconfig.device)

        seqs = k_prev_words

        top_k_scores = torch.zeros(k, 1).to(gconfig.device)

        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s <= k
        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1) # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h) # (s, encoder_dim)

            gate = decoder.sigmoid(decoder.f_beta(h)) # (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c)) # (s, decoder_dim)

            scores = decoder.fc(h) # (s, vocab_size)
            scores = torch.log_softmax(scores, dim=1)

            # Accumulate scores
            scores = top_k_scores.expand_as(scores) + scores # (s, vocab_size)

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True) # (s)

            prev_word_inds = top_k_words / vocab_size # (s)
            next_word_inds = top_k_words % vocab_size # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = seqs[0].tolist()

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    beam_size = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
